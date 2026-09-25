using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace UnityRhi.Dlss.Urp
{
    /// <summary>
    /// SDR DLSS Neural Rendering for Unity 2022.3 URP 14.
    /// Default injection is before post/upscale so the Game camera order is
    /// raster → NR → remaining post. XR uses per-eye history at native resolution
    /// (multipass or single-pass instanced texture arrays).
    /// </summary>
    [DisallowMultipleRendererFeature("DLSS Neural Rendering")]
    public sealed class DlssNrRenderFeature : ScriptableRendererFeature
    {
        [Tooltip("Default Before Rendering Post Processing runs NR at render resolution " +
                 "(NR → remaining post). After Rendering Post Processing is the " +
                 "optional post → NR stack.")]
        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        [Tooltip("Allow the effect in the Scene view when its Volume is active.")]
        public bool runInSceneView;

        [Tooltip("Only process the camera that resolves the final camera-stack target.")]
        public bool finalCameraInStackOnly = true;

        [SerializeField, Tooltip("Input preparation shader. Automatically resolved from the package when possible.")]
        private UnityEngine.Shader prepareInputsShader;

        private Material _prepareMaterial;
        private Material _debugMaterial;
        private Material _copyMaterial;
        private DlssNrPass _pass;
        private readonly Dictionary<long, DlssNrCameraContext> _contexts =
            new Dictionary<long, DlssNrCameraContext>();
        private readonly Dictionary<long, Camera> _contextCameras =
            new Dictionary<long, Camera>();
        private readonly List<long> _deadKeys = new List<long>();
        private bool _warnedUnavailable;
        private bool _warnedHdr;
        private bool _warnedFailure;
        private bool _loggedSrStack;
        private bool _loggedNrThenSr;
        private bool _loggedXr;
        private int _streamDiagFrames;

        public override void Create()
        {
            if (prepareInputsShader == null)
                prepareInputsShader = UnityEngine.Shader.Find("Hidden/UnityRHI/DLSS-NR/PrepareInputs");
            if (_prepareMaterial != null)
                CoreUtils.Destroy(_prepareMaterial);
            if (_debugMaterial != null)
                CoreUtils.Destroy(_debugMaterial);
            if (_copyMaterial != null)
                CoreUtils.Destroy(_copyMaterial);
            _prepareMaterial = prepareInputsShader != null
                ? CoreUtils.CreateEngineMaterial(prepareInputsShader) : null;
            // Separate draws bind their own material state. Independent materials
            // keep later pass bindings from mutating earlier recorded draws.
            _debugMaterial = prepareInputsShader != null
                ? CoreUtils.CreateEngineMaterial(prepareInputsShader) : null;
            _copyMaterial = prepareInputsShader != null
                ? CoreUtils.CreateEngineMaterial(prepareInputsShader) : null;
            _pass = new DlssNrPass(this) { renderPassEvent = renderPassEvent };
#if UNITY_EDITOR
            RhiDomainReload.RegisterOwner(this);
#endif
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (_prepareMaterial == null || _debugMaterial == null || _copyMaterial == null)
                return;
            DlssNrVolume volume = VolumeManager.instance.stack?.GetComponent<DlssNrVolume>();
            if (volume == null || !volume.IsActive())
                return;
            if (!RhiCore.IsD3D12Active || !RhiCore.IsDlssNrAvailable)
            {
                if (!_warnedUnavailable)
                {
                    _warnedUnavailable = true;
                    Debug.LogWarning($"[UnityRHI.DLSS-NR] Pass disabled: D3D12/NR runtime unavailable " +
                        $"(init=0x{unchecked((uint)RhiCore.DlssNrInitResult):X8}).");
                }
                return;
            }

            CameraData cameraData = renderingData.cameraData;
            Camera camera = cameraData.camera;
            if (camera == null || camera.cameraType == CameraType.Preview ||
                camera.cameraType == CameraType.Reflection)
                return;
            if (!runInSceneView && camera.cameraType == CameraType.SceneView)
                return;
            if (finalCameraInStackOnly && !cameraData.resolveFinalTarget)
                return;

            _pass.renderPassEvent = renderPassEvent;
            _pass.Renderer = renderer;
            _pass.ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Motion);
            renderer.EnqueuePass(_pass);
        }

        /// <summary>Reset temporal history for all live camera contexts.</summary>
        public void ResetHistory()
        {
            foreach (DlssNrCameraContext context in _contexts.Values)
                context.ResetHistory();
        }

        protected override void Dispose(bool disposing)
        {
            foreach (DlssNrCameraContext context in _contexts.Values)
                context.Dispose();
            _contexts.Clear();
            _contextCameras.Clear();
            _deadKeys.Clear();
            if (_prepareMaterial != null)
                CoreUtils.Destroy(_prepareMaterial);
            if (_debugMaterial != null)
                CoreUtils.Destroy(_debugMaterial);
            if (_copyMaterial != null)
                CoreUtils.Destroy(_copyMaterial);
            _prepareMaterial = null;
            _debugMaterial = null;
            _copyMaterial = null;
            _pass = null;
            base.Dispose(disposing);
        }

        private static long MakeContextKey(Camera camera, int eye) =>
            camera.GetInstanceID() + eye * 100000L;

        private DlssNrCameraContext GetContext(Camera camera, int eye, int width, int height,
            bool xr, int iterationCount)
        {
            PruneDeadCameras();
            long key = MakeContextKey(camera, eye);
            if (_contexts.TryGetValue(key, out DlssNrCameraContext context))
            {
                if (context.Width == width && context.Height == height &&
                    context.IterationCount == iterationCount)
                    return context;
                context.Dispose();
                _contexts.Remove(key);
                _contextCameras.Remove(key);
            }

            string name = xr ? $"{camera.name}_Eye{eye}" : camera.name;
            context = new DlssNrCameraContext(width, height, name, iterationCount);
            _contexts.Add(key, context);
            _contextCameras[key] = camera;
            return context;
        }

        private void PruneDeadCameras()
        {
            _deadKeys.Clear();
            foreach (KeyValuePair<long, Camera> pair in _contextCameras)
                if (pair.Value == null)
                    _deadKeys.Add(pair.Key);
            foreach (long key in _deadKeys)
            {
                if (_contexts.TryGetValue(key, out DlssNrCameraContext context))
                    context.Dispose();
                _contexts.Remove(key);
                _contextCameras.Remove(key);
            }
            _deadKeys.Clear();
        }

        private sealed class DlssNrPass : ScriptableRenderPass
        {
            private static readonly int InputColorId = UnityEngine.Shader.PropertyToID("_DlssNrInputColor");
            private static readonly int InputDepthId = UnityEngine.Shader.PropertyToID("_DlssNrInputDepth");
            private static readonly int InputMotionId = UnityEngine.Shader.PropertyToID("_DlssNrInputMotion");
            private static readonly int EyeSliceId = UnityEngine.Shader.PropertyToID("_DlssNrEyeSlice");
            private static readonly int CopySourceId = UnityEngine.Shader.PropertyToID("_DlssNrCopySource");
            private static readonly int DebugModeId = UnityEngine.Shader.PropertyToID("_DlssNrDebugMode");
            private static readonly int DebugMotionScaleXId = UnityEngine.Shader.PropertyToID("_DlssNrDebugMotionScaleX");
            private static readonly int DebugMotionScaleYId = UnityEngine.Shader.PropertyToID("_DlssNrDebugMotionScaleY");
            private static readonly int DebugMotionRangeId = UnityEngine.Shader.PropertyToID("_DlssNrDebugMotionRange");
            private static readonly int DebugDepthRangeId = UnityEngine.Shader.PropertyToID("_DlssNrDebugDepthRange");
            // URP-bound global textures. ConfigureInput(Depth | Motion) guarantees
            // URP resolves and binds these before this pass runs.
            private static readonly int CameraDepthTextureId =
                UnityEngine.Shader.PropertyToID("_CameraDepthTexture");
            private static readonly int MotionVectorTextureId =
                UnityEngine.Shader.PropertyToID("_MotionVectorTexture");
            private static readonly string ColorArrayKeyword = "_DLSSNR_COLOR_ARRAY";
            private static readonly string DepthArrayKeyword = "_DLSSNR_DEPTH_ARRAY";
            private static readonly string MotionArrayKeyword = "_DLSSNR_MOTION_ARRAY";
            private readonly DlssNrRenderFeature _feature;
            private readonly MaterialPropertyBlock _properties = new MaterialPropertyBlock();

            /// <summary>Renderer whose color target is the NR input/output.</summary>
            internal ScriptableRenderer Renderer { get; set; }

            internal DlssNrPass(DlssNrRenderFeature feature)
            {
                _feature = feature;
                profilingSampler = new ProfilingSampler("DLSS Neural Rendering");
            }

            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                if (_feature._prepareMaterial == null || _feature._debugMaterial == null ||
                    _feature._copyMaterial == null || Renderer == null)
                    return;

                ref CameraData cameraData = ref renderingData.cameraData;
                Camera camera = cameraData.camera;
                if (camera == null)
                    return;
                if (_feature.finalCameraInStackOnly && !cameraData.resolveFinalTarget)
                    return;
                DlssNrVolume volume = VolumeManager.instance.stack?.GetComponent<DlssNrVolume>();
                if (volume == null || !volume.IsActive())
                    return;
                var settings = new DlssNrSettings(volume);
                if (cameraData.isHDROutputActive)
                {
                    if (!_feature._warnedHdr)
                    {
                        _feature._warnedHdr = true;
                        Debug.LogWarning("[UnityRHI.DLSS-NR] HDR Output is not supported on this SDR path; bypassing.");
                    }
                    return;
                }

                RTHandle sourceColor = Renderer.cameraColorTargetHandle;
                var sourceDepth = UnityEngine.Shader.GetGlobalTexture(CameraDepthTextureId) as RenderTexture;
                var sourceMotion = UnityEngine.Shader.GetGlobalTexture(MotionVectorTextureId) as RenderTexture;
                if (sourceColor == null || sourceColor.rt == null ||
                    sourceDepth == null || sourceMotion == null)
                    return;

                RenderTexture sourceColorRt = sourceColor.rt;
                int width = sourceColorRt.width;
                int height = sourceColorRt.height;
                if (width <= 0 || height <= 0)
                    return;

                bool auxLowerRes = sourceDepth.width < width || sourceDepth.height < height ||
                    sourceMotion.width < width || sourceMotion.height < height;
                if (auxLowerRes && !_feature._loggedSrStack)
                {
                    _feature._loggedSrStack = true;
                    Debug.Log(
                        $"[UnityRHI.DLSS-NR] SR→NR stack: color {width}x{height}, " +
                        $"depth {sourceDepth.width}x{sourceDepth.height}, " +
                        $"motion {sourceMotion.width}x{sourceMotion.height}. " +
                        "Upsampling depth/MV into NR inputs.");
                }
                else if (!auxLowerRes && !_feature._loggedNrThenSr)
                {
                    _feature._loggedNrThenSr = true;
                    Debug.Log(
                        $"[UnityRHI.DLSS-NR] NR at {width}x{height} " +
                        "(native / render resolution).");
                }

                ResolveEyes(cameraData, sourceColorRt, out int firstEye, out int eyeCount,
                    out bool texArray);
                if (texArray && !_feature._loggedXr)
                {
                    _feature._loggedXr = true;
                    Debug.Log(
                        $"[UnityRHI.DLSS-NR] XR single-pass array: {eyeCount} eyes at {width}x{height}.");
                }
                else if (cameraData.xr.enabled && !texArray && !_feature._loggedXr)
                {
                    _feature._loggedXr = true;
                    Debug.Log(
                        $"[UnityRHI.DLSS-NR] XR multipass eye {firstEye} at {width}x{height}.");
                }

                bool colorArray = IsTexArray(sourceColorRt);
                bool depthArray = IsTexArray(sourceDepth);
                bool motionArray = IsTexArray(sourceMotion);

                CommandBuffer cmd = CommandBufferPool.Get();
                using (new ProfilingScope(cmd, profilingSampler))
                {
                    for (int i = 0; i < eyeCount; ++i)
                    {
                        int eye = firstEye + i;
                        if (!TryRecordEye(cmd, cameraData, camera, settings,
                                sourceColor, sourceDepth, sourceMotion, width, height,
                                eye, texArray, colorArray, depthArray, motionArray))
                        {
                            CommandBufferPool.Release(cmd);
                            return;
                        }
                    }
                }

                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);
            }

            private static bool IsTexArray(RenderTexture texture) =>
                texture.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray &&
                texture.volumeDepth > 1;

            private static void ResolveEyes(CameraData cameraData, RenderTexture sourceColor,
                out int firstEye, out int eyeCount, out bool texArray)
            {
                // NativeNrd keys history by cameraId + multipassId * 100000. Single-pass
                // instanced still uses that key scheme, but NGX only accepts 2D resources,
                // so we extract each array slice into a per-eye 2D context.
                texArray = false;
                if (cameraData.xr.enabled && cameraData.xr.singlePassEnabled &&
                    IsTexArray(sourceColor))
                {
                    firstEye = 0;
                    eyeCount = Mathf.Max(1, cameraData.xr.viewCount);
                    texArray = true;
                    return;
                }

                firstEye = cameraData.xr.enabled ? cameraData.xr.multipassId : 0;
                eyeCount = 1;
            }

            private bool TryRecordEye(CommandBuffer cmd,
                CameraData cameraData, Camera camera, in DlssNrSettings settings,
                RTHandle sourceColor, RenderTexture sourceDepth, RenderTexture sourceMotion,
                int width, int height, int eye, bool texArray,
                bool colorArray, bool depthArray, bool motionArray)
            {
                DlssNrCameraContext ctx;
                try
                {
                    ctx = _feature.GetContext(camera, eye, width, height,
                        cameraData.xr.enabled, settings.DebugMode == DlssNrDebugMode.Off
                            ? settings.IterationCount : 1);
                }
                catch (Exception exception)
                {
                    if (!_feature._warnedFailure)
                    {
                        _feature._warnedFailure = true;
                        Debug.LogError($"[UnityRHI.DLSS-NR] Resource initialization failed; bypassing. {exception}");
                    }
                    return false;
                }

                // Prepare: copy color/depth/motion into the persistent NGX inputs.
                // MRT order matches the shader's SV_Target0=color, 1=motion, 2=depth.
                // The shader's SV_Target3 fallback is intentionally left unbound: the
                // native stream owns OutputRt (UAV). Binding it as a render attachment
                // here would make Unity's state tracker record OutputRt as RenderTarget
                // and the later Blit would read black (native leaves it UnorderedAccess).
                SetKeyword(_feature._prepareMaterial, ColorArrayKeyword, colorArray);
                SetKeyword(_feature._prepareMaterial, DepthArrayKeyword, depthArray);
                SetKeyword(_feature._prepareMaterial, MotionArrayKeyword, motionArray);
                var prepareTargets = new RenderTargetIdentifier[]
                {
                    ctx.ColorHandle.nameID,
                    ctx.MotionHandle.nameID,
                    ctx.DepthHandle.nameID,
                };
                cmd.SetRenderTarget(prepareTargets, ctx.ColorHandle.nameID);
                cmd.SetViewport(new Rect(0f, 0f, width, height));
                _properties.Clear();
                _properties.SetTexture(InputColorId, sourceColor);
                _properties.SetTexture(InputDepthId, sourceDepth);
                _properties.SetTexture(InputMotionId, sourceMotion);
                _properties.SetFloat(EyeSliceId, eye);
                cmd.DrawProcedural(Matrix4x4.identity, _feature._prepareMaterial, 0,
                    MeshTopology.Triangles, 3, 1, _properties);

                GetEyePose(cameraData, camera, eye,
                    out Vector3 position, out Quaternion rotation, out Matrix4x4 projection);

                if (settings.DebugMode != DlssNrDebugMode.Off)
                {
                    cmd.SetRenderTarget(ctx.OutputHandle.nameID);
                    cmd.SetViewport(new Rect(0f, 0f, width, height));
                    _properties.Clear();
                    _properties.SetTexture(InputDepthId, ctx.DepthHandle);
                    _properties.SetTexture(InputMotionId, ctx.MotionHandle);
                    _properties.SetInt(DebugModeId, (int)settings.DebugMode);
                    _properties.SetFloat(DebugMotionScaleXId, -width * settings.MotionVectorScale.x);
                    _properties.SetFloat(DebugMotionScaleYId, -height * settings.MotionVectorScale.y);
                    _properties.SetFloat(DebugMotionRangeId, settings.DebugMotionRange);
                    _properties.SetFloat(DebugDepthRangeId, settings.DebugDepthRange);
                    cmd.DrawProcedural(Matrix4x4.identity, _feature._debugMaterial, 1,
                        MeshTopology.Triangles, 3, 1, _properties);

                    // Debug output is written by Unity raster into OutputHandle.
                    ResolveOutput(cmd, ctx.OutputHandle, sourceColor, texArray, eye, width, height);
                    return true;
                }

                DlssNrCameraContext.DispatchParameters parameters =
                    ctx.BeginFrame(Time.frameCount, position, rotation, projection, settings);
                try
                {
                    // The native NR stream reads the prepared inputs, writes OutputRt (UAV)
                    // and copies it into ResolveRt, restoring ResolveRt to RenderTarget.
                    ctx.Record(cmd, parameters);
                }
                catch (Exception exception)
                {
                    if (!_feature._warnedFailure)
                    {
                        _feature._warnedFailure = true;
                        Debug.LogError($"[UnityRHI.DLSS-NR] Dispatch failed. {exception}");
                    }
                    return false;
                }

                // The native result is read back from ResolveHandle (see ctx.Record).
                ResolveOutput(cmd, ctx.ResolveHandle, sourceColor, texArray, eye, width, height);
                return true;
            }

            // Move the NR result back onto the camera color. For single 2D targets a
            // full blit; for XR texture arrays a per-eye slice copy.
            // The debug path passes OutputHandle (written by Unity raster). The native
            // path passes ResolveHandle (the native stream's copy of OutputRt), whose
            // state Unity can track; blitting OutputHandle after a native write is black.
            private void ResolveOutput(CommandBuffer cmd, RTHandle source,
                RTHandle destination, bool texArray, int eye, int width, int height)
            {
                if (!texArray)
                {
                    Blitter.BlitCameraTexture(cmd, source, destination);
                    return;
                }

                cmd.SetRenderTarget(destination.nameID, 0, CubemapFace.Unknown, eye);
                cmd.SetViewport(new Rect(0f, 0f, width, height));
                _properties.Clear();
                _properties.SetTexture(CopySourceId, source);
                cmd.DrawProcedural(Matrix4x4.identity, _feature._copyMaterial, 2,
                    MeshTopology.Triangles, 3, 1, _properties);
            }

            private static void SetKeyword(Material material, string keyword, bool enabled)
            {
                if (enabled)
                    material.EnableKeyword(keyword);
                else
                    material.DisableKeyword(keyword);
            }

            private static void GetEyePose(CameraData cameraData, Camera camera,
                int viewIndex, out Vector3 position, out Quaternion rotation,
                out Matrix4x4 projection)
            {
                if (cameraData.xr.enabled)
                {
                    int index = Mathf.Clamp(viewIndex, 0, Mathf.Max(0, cameraData.xr.viewCount - 1));
                    Matrix4x4 invView = cameraData.xr.GetViewMatrix(index).inverse;
                    position = invView.GetColumn(3);
                    rotation = invView.rotation;
                    projection = cameraData.xr.GetProjMatrix(index);
                    return;
                }

                position = camera.transform.position;
                rotation = camera.transform.rotation;
                projection = camera.nonJitteredProjectionMatrix;
            }
        }
    }
}
