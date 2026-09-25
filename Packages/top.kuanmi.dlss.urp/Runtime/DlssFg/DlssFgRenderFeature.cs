using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace UnityRhi.Dlss.Urp
{
    public enum DlssFgDebugView
    {
        Off,
        DeviceDepth,
        LinearDepth,
        MotionVectors,
        CameraMotion,
        MotionError,
        MotionComparison,
    }

    /// <summary>
    /// Copies URP depth and motion into persistent textures and submits them to
    /// the native DLSS-G Present path. Color is taken from the swap chain at Present.
    /// Player-only; the Editor has no Present hook that can insert generated frames.
    /// </summary>
    [DisallowMultipleRendererFeature("DLSS Frame Generation")]
    public sealed class DlssFgRenderFeature : ScriptableRendererFeature
    {
        [Tooltip("Late injection copies depth/motion after post and upscale. " +
                 "The interpolated color comes from the swap chain at Present.")]
        public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing;

        [Tooltip("Evaluate DLSS-G and insert generated frames. Player-only.")]
        public bool enableFrameGeneration = true;

        [Tooltip("Only process the camera that resolves the final camera-stack target.")]
        public bool finalCameraInStackOnly = true;

        [Tooltip("Camera movement larger than this in one rendered frame resets temporal history.")]
        public float cameraCutDistance = 5f;

        [Tooltip("Camera rotation larger than this in one rendered frame resets temporal history.")]
        public float cameraCutAngle = 45f;

        [Tooltip("Editor/Game view used to validate the exact depth, motion and camera matrices submitted to DLSS-G.")]
        public DlssFgDebugView debugView = DlssFgDebugView.Off;

        [Min(1f), Tooltip("Pixel-motion magnitude mapped to the edge of the motion debug color range.")]
        public float debugMotionRangePixels = 32f;

        [SerializeField, Tooltip("Input preparation shader. Automatically resolved from the package when possible.")]
        private UnityEngine.Shader prepareInputsShader;

        private static DlssFgRenderFeature s_live;
        private bool? _runtimeEnabled;
        private Material _prepareMaterial;
        private DlssFgPass _pass;
        private readonly Dictionary<int, DlssFgCameraContext> _contexts =
            new Dictionary<int, DlssFgCameraContext>();
        private readonly Dictionary<int, Camera> _contextCameras =
            new Dictionary<int, Camera>();
        private readonly List<int> _deadKeys = new List<int>();
        private bool _warnedUnavailable;
        private bool _warnedFailure;
        private bool _pacingOverridden;
        private int _savedVSync;
        private int _savedTargetFrameRate;
        private string _lastSkipReason;

        public override void Create()
        {
            if (prepareInputsShader == null)
                prepareInputsShader = UnityEngine.Shader.Find("Hidden/UnityRHI/DLSS-G/PrepareInputs");
            if (_prepareMaterial != null)
                CoreUtils.Destroy(_prepareMaterial);
            _prepareMaterial = prepareInputsShader != null
                ? CoreUtils.CreateEngineMaterial(prepareInputsShader) : null;
            _pass = new DlssFgPass(this) { renderPassEvent = renderPassEvent };
            s_live = this;
            Debug.Log(
                $"[UnityRHI.DLSS-G] Feature Create active={isActive} enable={enableFrameGeneration} " +
                $"shader={(_prepareMaterial != null)} editor={Application.isEditor}");
#if UNITY_EDITOR
            RhiDomainReload.RegisterOwner(this);
#endif
        }

        /// <summary>True when the live renderer feature will submit DLSS-G this frame.</summary>
        public static bool IsFrameGenerationEnabled =>
            s_live != null && s_live.isActive && s_live.RuntimeEnabled;

        private bool RuntimeEnabled => _runtimeEnabled ?? enableFrameGeneration;

        /// <summary>Runtime toggle for external callers. Does not require a Volume.</summary>
        public static void SetFrameGenerationEnabled(bool enabled)
        {
            if (s_live == null)
            {
                Debug.LogWarning("[UnityRHI.DLSS-G] Toggle ignored: renderer feature is not live.");
                return;
            }

            bool before = s_live.RuntimeEnabled;
            s_live._runtimeEnabled = enabled;
            Debug.Log(
                $"[UnityRHI.DLSS-G] SetFrameGenerationEnabled {before} -> {enabled} " +
                $"active={s_live.isActive} inspectorEnable={s_live.enableFrameGeneration} " +
                $"editor={Application.isEditor} ngx={RhiCore.IsNgxFrameGenerationAvailable}");
            if (!enabled || Application.isEditor)
                s_live.DisableFrameGeneration();
            else
                RhiCore.SetFrameGenerationEnabled(true);
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            bool debugActive = debugView != DlssFgDebugView.Off;
            bool canSubmit = !Application.isEditor && RuntimeEnabled &&
                RhiCore.IsD3D12Active && RhiCore.IsNgxFrameGenerationAvailable;
            string skip = null;
            if (_prepareMaterial == null)
                skip = "missing-shader";
            else if (Application.isEditor && !debugActive)
                skip = "editor";
            else if (!Application.isEditor && !RuntimeEnabled && !debugActive)
                skip = "disabled";
            else if (!Application.isEditor && !RhiCore.IsD3D12Active && !debugActive)
                skip = "not-d3d12";
            else if (!Application.isEditor && !RhiCore.IsNgxFrameGenerationAvailable && !debugActive)
                skip = $"ngx-unavailable:0x{unchecked((uint)RhiCore.NgxFrameGenerationInitResult):X8}";
            else if (!IsSupportedCamera(renderingData.cameraData))
                skip = "unsupported-camera";

            if (skip != null)
            {
                LogSkip(skip, renderingData.cameraData);
                if (skip != "unsupported-camera")
                    DisableFrameGeneration();
                return;
            }

            LogSkip(null, renderingData.cameraData);
            if (canSubmit)
                ApplyPacing(true);
            else
                DisableFrameGeneration();
            _pass.renderPassEvent = renderPassEvent;
            _pass.Renderer = renderer;
            _pass.ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Motion);
            renderer.EnqueuePass(_pass);
        }

        private void LogSkip(string reason, CameraData cameraData)
        {
            Camera camera = cameraData.camera;
            string key = reason ?? "enqueue";
            if (key == _lastSkipReason)
                return;
            _lastSkipReason = key;
            string cameraName = camera != null ? camera.name : "null";
            Debug.Log(
                $"[UnityRHI.DLSS-G] Pass {key} camera={cameraName} type={camera?.cameraType} " +
                $"runtimeEnabled={RuntimeEnabled} active={isActive}");
            if (reason == "ngx-unavailable" || (reason != null && reason.StartsWith("ngx-unavailable")))
            {
                if (!_warnedUnavailable)
                {
                    _warnedUnavailable = true;
                    Debug.LogWarning(
                        $"[UnityRHI.DLSS-G] Pass disabled: D3D12/FG runtime unavailable " +
                        $"(init=0x{unchecked((uint)RhiCore.NgxFrameGenerationInitResult):X8}).");
                }
            }
        }

        /// <summary>Reset temporal history for all live camera contexts.</summary>
        public void ResetHistory()
        {
            foreach (DlssFgCameraContext context in _contexts.Values)
                context.ResetHistory();
        }

        protected override void Dispose(bool disposing)
        {
#if UNITY_EDITOR
            RhiDomainReload.UnregisterOwner(this);
#endif
            if (ReferenceEquals(s_live, this))
                s_live = null;
            DisableFrameGeneration();
            foreach (DlssFgCameraContext context in _contexts.Values)
                context.Dispose();
            _contexts.Clear();
            _contextCameras.Clear();
            _deadKeys.Clear();
            if (_prepareMaterial != null)
                CoreUtils.Destroy(_prepareMaterial);
            _prepareMaterial = null;
            _pass = null;
            base.Dispose(disposing);
        }

        private bool IsSupportedCamera(CameraData cameraData)
        {
            Camera camera = cameraData.camera;
            if (camera == null || camera.cameraType != CameraType.Game || camera.targetTexture != null)
                return false;
            if (finalCameraInStackOnly && !cameraData.resolveFinalTarget)
                return false;
            return !cameraData.xr.enabled;
        }

        private void DisableFrameGeneration()
        {
            RhiCore.SetFrameGenerationEnabled(false);
            ApplyPacing(false);
        }

        private void ApplyPacing(bool enabled)
        {
            if (enabled)
            {
                if (!_pacingOverridden)
                {
                    _savedVSync = QualitySettings.vSyncCount;
                    _savedTargetFrameRate = Application.targetFrameRate;
                    _pacingOverridden = true;
                }
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = -1;
                return;
            }

            if (!_pacingOverridden)
                return;
            QualitySettings.vSyncCount = _savedVSync;
            Application.targetFrameRate = _savedTargetFrameRate;
            _pacingOverridden = false;
        }

        private DlssFgCameraContext GetContext(Camera camera, int width, int height)
        {
            PruneDeadCameras();
            int key = camera.GetInstanceID();
            if (_contexts.TryGetValue(key, out DlssFgCameraContext context))
            {
                if (context.Width == width && context.Height == height)
                    return context;
                context.Dispose();
                _contexts.Remove(key);
                _contextCameras.Remove(key);
            }

            context = new DlssFgCameraContext(width, height, camera.name);
            _contexts.Add(key, context);
            _contextCameras[key] = camera;
            return context;
        }

        private void PruneDeadCameras()
        {
            _deadKeys.Clear();
            foreach (KeyValuePair<int, Camera> pair in _contextCameras)
                if (pair.Value == null)
                    _deadKeys.Add(pair.Key);
            foreach (int key in _deadKeys)
            {
                if (_contexts.TryGetValue(key, out DlssFgCameraContext context))
                    context.Dispose();
                _contexts.Remove(key);
                _contextCameras.Remove(key);
            }
            _deadKeys.Clear();
        }

        private sealed class DlssFgPass : ScriptableRenderPass
        {
            private static readonly int InputDepthId = UnityEngine.Shader.PropertyToID("_DlssFgInputDepth");
            private static readonly int InputMotionId = UnityEngine.Shader.PropertyToID("_DlssFgInputMotion");
            private static readonly int DepthScaleBiasId =
                UnityEngine.Shader.PropertyToID("_DlssFgDepthScaleBias");
            private static readonly int MotionScaleBiasId =
                UnityEngine.Shader.PropertyToID("_DlssFgMotionScaleBias");
            private static readonly int DebugDepthId = UnityEngine.Shader.PropertyToID("_DlssFgDebugDepth");
            private static readonly int DebugMotionId = UnityEngine.Shader.PropertyToID("_DlssFgDebugMotion");
            private static readonly int DebugClipToPrevClipId =
                UnityEngine.Shader.PropertyToID("_DlssFgDebugClipToPrevClip");
            private static readonly int DebugSizeId = UnityEngine.Shader.PropertyToID("_DlssFgDebugSize");
            private static readonly int DebugParamsId = UnityEngine.Shader.PropertyToID("_DlssFgDebugParams");
            private static readonly int DebugMotionScaleId =
                UnityEngine.Shader.PropertyToID("_DlssFgDebugMotionScale");
            // URP-bound global textures. ConfigureInput(Depth | Motion) guarantees
            // URP resolves and binds these before this pass runs.
            private static readonly int CameraDepthTextureId =
                UnityEngine.Shader.PropertyToID("_CameraDepthTexture");
            private static readonly int MotionVectorTextureId =
                UnityEngine.Shader.PropertyToID("_MotionVectorTexture");

            private readonly DlssFgRenderFeature _feature;
            private readonly MaterialPropertyBlock _properties = new MaterialPropertyBlock();

            /// <summary>Renderer whose color target backs the debug view.</summary>
            internal ScriptableRenderer Renderer { get; set; }

            // A per-camera snapshot passed to the native submission after the copy.
            private sealed class SubmitData
            {
                public DlssFgCameraContext Context;
                public Matrix4x4 ViewToClip;
                public Matrix4x4 ViewProj;
                public Matrix4x4 PrevViewProj;
                public Vector3 Position;
                public Vector3 Up;
                public Vector3 Right;
                public Vector3 Forward;
                public Vector2 JitterPixels;
                public Vector2 MotionScale;
                public float Near;
                public float Far;
                public float Fov;
                public float Aspect;
                public int ColorWidth;
                public int ColorHeight;
                public int FrameIndex;
                public bool Reset;
                public bool ColorBuffersHdr;
            }

            private static bool s_loggedSubmit;

            internal DlssFgPass(DlssFgRenderFeature feature)
            {
                _feature = feature;
                profilingSampler = new ProfilingSampler("DLSS Frame Generation");
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                if (_feature._prepareMaterial == null)
                    return;

                ref CameraData cameraData = ref renderingData.cameraData;
                Camera camera = cameraData.camera;
                if (camera == null)
                    return;
                if (_feature.finalCameraInStackOnly && !cameraData.resolveFinalTarget)
                    return;
                bool debugActive = _feature.debugView != DlssFgDebugView.Off;
                if (!_feature.RuntimeEnabled && !debugActive)
                    return;
                if (cameraData.xr.enabled)
                    return;

                // URP publishes depth and motion as global textures once
                // ConfigureInput requests them. NGX consumes copies of both.
                var sourceDepth = UnityEngine.Shader.GetGlobalTexture(CameraDepthTextureId) as RenderTexture;
                var sourceMotion = UnityEngine.Shader.GetGlobalTexture(MotionVectorTextureId) as RenderTexture;
                if (sourceDepth == null || sourceMotion == null)
                    return;

                int width = Mathf.Min(sourceDepth.width, sourceMotion.width);
                int height = Mathf.Min(sourceDepth.height, sourceMotion.height);
                if (width <= 0 || height <= 0)
                    return;
                if (IsTexArray(sourceDepth) || IsTexArray(sourceMotion))
                    return;

                var settings = new DlssFgSettings(_feature);
                DlssFgCameraContext ctx;
                try
                {
                    ctx = _feature.GetContext(camera, width, height);
                }
                catch (Exception exception)
                {
                    if (!_feature._warnedFailure)
                    {
                        _feature._warnedFailure = true;
                        Debug.LogError($"[UnityRHI.DLSS-G] Resource initialization failed; bypassing. {exception}");
                    }
                    return;
                }

                Matrix4x4 view = cameraData.GetViewMatrix();
                // URP's MotionVectorsPersistentData uses camera.projectionMatrix as
                // the no-jitter base, then applies cameraData's jitter separately.
                // Use that exact base so ClipToPrevClip describes sourceMotion.
                Matrix4x4 projection = camera.projectionMatrix;
                // Match URP's overlay viewport adjustment as well as its base
                // projection. camera.nonJitteredProjectionMatrix is not URP's base.
                if (cameraData.renderType == CameraRenderType.Overlay && !camera.orthographic)
                    projection.m00 = cameraData.GetProjectionMatrix().m00;
                // Depth and motion are copied into NGX's canonical top-left
                // texture space below, so the submitted camera matrices must
                // use the matching non-render-texture clip-space orientation.
                Matrix4x4 viewToClip = GL.GetGPUProjectionMatrix(projection, false);
                Matrix4x4 viewProj = viewToClip * view;
                Vector3 position = camera.transform.position;
                Quaternion rotation = camera.transform.rotation;
                ctx.BeginFrame(Time.frameCount, position, rotation, projection, viewProj,
                    settings, out Matrix4x4 prevViewProj, out bool reset);
                Vector2 jitterPixels = ExtractJitterPixels(cameraData, projection, width, height);

                bool submitToNgx = !Application.isEditor && _feature.RuntimeEnabled &&
                    RhiCore.IsD3D12Active && RhiCore.IsNgxFrameGenerationAvailable;

                CommandBuffer cmd = CommandBufferPool.Get();
                using (new ProfilingScope(cmd, profilingSampler))
                {
                    // Prepare: copy URP depth/motion into the persistent NGX inputs.
                    // MRT order matches the shader's SV_Target0=motion, SV_Target1=depth.
                    var targets = new RenderTargetIdentifier[]
                    {
                        ctx.MotionHandle.nameID,
                        ctx.DepthHandle.nameID,
                    };
                    // These inputs carry no depth-stencil surface; reuse a color
                    // attachment as the (unused, ZTest Always / ZWrite Off) depth slot.
                    cmd.SetRenderTarget(targets, ctx.MotionHandle.nameID);
                    cmd.SetViewport(new Rect(0f, 0f, width, height));

                    _properties.Clear();
                    _properties.SetTexture(InputDepthId, sourceDepth);
                    _properties.SetTexture(InputMotionId, sourceMotion);
                    _properties.SetVector(DepthScaleBiasId, GetNgxTextureScaleBias());
                    _properties.SetVector(MotionScaleBiasId, GetNgxTextureScaleBias());
                    cmd.DrawProcedural(Matrix4x4.identity, _feature._prepareMaterial, 0,
                        MeshTopology.Triangles, 3, 1, _properties);

                    if (debugActive && Renderer != null && Renderer.cameraColorTargetHandle != null)
                    {
                        Matrix4x4 clipToPrevClip = prevViewProj * viewProj.inverse;
                        CoreUtils.SetRenderTarget(cmd, Renderer.cameraColorTargetHandle);
                        _properties.Clear();
                        _properties.SetTexture(DebugDepthId, ctx.DepthHandle);
                        _properties.SetTexture(DebugMotionId, ctx.MotionHandle);
                        _properties.SetMatrix(DebugClipToPrevClipId, clipToPrevClip);
                        _properties.SetVector(DebugSizeId,
                            new Vector4(width, height, 1f / width, 1f / height));
                        _properties.SetVector(DebugParamsId, new Vector4((float)_feature.debugView,
                            SystemInfo.usesReversedZBuffer ? 1f : 0f,
                            Mathf.Max(1f, _feature.debugMotionRangePixels),
                            Mathf.Max(1e-4f, camera.nearClipPlane)));
                        _properties.SetVector(DebugMotionScaleId, new Vector4(-width, -height, 0f, 0f));
                        cmd.DrawProcedural(Matrix4x4.identity, _feature._prepareMaterial, 1,
                            MeshTopology.Triangles, 3, 1, _properties);
                    }

                    if (submitToNgx)
                    {
                        // Submit after the copy so NGX reads the finished SRVs.
                        var submit = new SubmitData
                        {
                            Context = ctx,
                            ViewToClip = viewToClip,
                            ViewProj = viewProj,
                            PrevViewProj = prevViewProj,
                            Position = position,
                            Up = camera.transform.up,
                            Right = camera.transform.right,
                            Forward = camera.transform.forward,
                            JitterPixels = jitterPixels,
                            // DLSS-G consumes current-to-previous motion in pixels. URP
                            // stores previous-to-current motion in normalized screen UV.
                            MotionScale = new Vector2(-width, -height),
                            Near = Mathf.Max(1e-4f, camera.nearClipPlane),
                            Far = camera.farClipPlane,
                            Fov = camera.fieldOfView * Mathf.Deg2Rad,
                            Aspect = (float)width / Mathf.Max(1, height),
                            ColorWidth = Mathf.Max(1, Screen.width),
                            ColorHeight = Mathf.Max(1, Screen.height),
                            FrameIndex = Time.frameCount,
                            Reset = reset,
                            ColorBuffersHdr = cameraData.isHDROutputActive,
                        };
                        SubmitInputs(submit, cmd);
                    }
                }

                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
            }

            private static Vector4 GetNgxTextureScaleBias()
            {
                // NGX consumes raw D3D resources using a top-left origin. On D3D12
                // (the only supported backend) sampling URP's global depth/motion
                // with a full-screen triangle already lands in top-left space, so
                // no vertical flip is needed. Keep the flipped branch as a guard
                // for backends whose logical textures start at the bottom-left.
                return SystemInfo.graphicsUVStartsAtTop
                    ? new Vector4(1f, 1f, 0f, 0f)
                    : new Vector4(1f, -1f, 0f, 1f);
            }

            private static bool IsTexArray(RenderTexture texture) =>
                texture.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray &&
                texture.volumeDepth > 1;

            private static Vector2 ExtractJitterPixels(CameraData cameraData,
                in Matrix4x4 unjitteredProjection, int width, int height)
            {
                Matrix4x4 jittered = cameraData.GetProjectionMatrix();
                Matrix4x4 translation = jittered * unjitteredProjection.inverse;
                return new Vector2(
                    // The rasterized image shifts by (+x,+y) in URP's flipped
                    // RT, then the preparation copy reverses Y. Sampling offsets
                    // are the opposite image displacement: (-x,+y) in NGX space.
                    -translation.m03 * 0.5f * width,
                    translation.m13 * 0.5f * height);
            }

            private static void SubmitInputs(SubmitData data, CommandBuffer commandBuffer)
            {
                IntPtr depth = data.Context.DepthRt.GetNativeTexturePtr();
                IntPtr motion = data.Context.MotionRt.GetNativeTexturePtr();
                if (depth == IntPtr.Zero || motion == IntPtr.Zero)
                {
                    Debug.LogWarning("[UnityRHI.DLSS-G] Submit skipped: native depth/motion pointer is zero.");
                    return;
                }

                if (!s_loggedSubmit)
                {
                    s_loggedSubmit = true;
                    Debug.Log(
                        $"[UnityRHI.DLSS-G] First SubmitInputs frame={data.FrameIndex} " +
                        $"render={data.Context.Width}x{data.Context.Height} " +
                        $"color={data.ColorWidth}x{data.ColorHeight} reset={data.Reset}");
                }

                Matrix4x4 clipToPrevClip = data.PrevViewProj * data.ViewProj.inverse;
                Matrix4x4 prevClipToClip = data.ViewProj * data.PrevViewProj.inverse;
                var inputs = new FrameGenerationInputs
                {
                    Depth = depth,
                    MotionVectors = motion,
                    RenderWidth = (uint)data.Context.Width,
                    RenderHeight = (uint)data.Context.Height,
                    ColorWidth = (uint)data.ColorWidth,
                    ColorHeight = (uint)data.ColorHeight,
                    DepthState = FrameGenerationInputs.D3D12ResourceStateNonPixelShaderResource,
                    MotionVectorsState = FrameGenerationInputs.D3D12ResourceStateNonPixelShaderResource,
                    FrameId = unchecked((ulong)Mathf.Max(0, data.FrameIndex)),
                    CameraViewToClip = FrameGenerationMatrix.FromUnity(data.ViewToClip),
                    ClipToCameraView = FrameGenerationMatrix.FromUnity(data.ViewToClip.inverse),
                    ClipToPrevClip = FrameGenerationMatrix.FromUnity(clipToPrevClip),
                    PrevClipToClip = FrameGenerationMatrix.FromUnity(prevClipToClip),
                    CameraPos = data.Position,
                    CameraUp = data.Up,
                    CameraRight = data.Right,
                    CameraFwd = data.Forward,
                    JitterX = data.JitterPixels.x,
                    JitterY = data.JitterPixels.y,
                    MotionVectorScaleX = data.MotionScale.x,
                    MotionVectorScaleY = data.MotionScale.y,
                    CameraNear = data.Near,
                    CameraFar = data.Far,
                    CameraFov = data.Fov,
                    CameraAspect = data.Aspect,
                    DepthInverted = SystemInfo.usesReversedZBuffer ? 1u : 0u,
                    CameraMotionIncluded = 1,
                    Reset = data.Reset ? 1u : 0u,
                    ColorBuffersHdr = data.ColorBuffersHdr ? 1u : 0u,
                };
                RhiCore.SetFrameGenerationEnabled(true);
                RhiCore.SubmitFrameGenerationInputs(commandBuffer, in inputs);
            }
        }
    }
}
