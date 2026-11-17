# Polyfitty Studio

<p align="center">
  <img src="app/assets/PolyfittyStudio.png" alt="Polyfitty Studio logo" width="180">
</p>

<p align="center">
  <strong>A Windows desktop workspace for 3D surface reconstruction, mesh inspection, texture baking, and asset export.</strong>
</p>

<p align="center">
  <code>Version 0.2.131</code> · <code>Windows x64</code> · <code>GNU GPL-3.0</code>
</p>

Polyfitty Studio turns point clouds and existing 3D meshes into clean, inspectable polygonal surfaces. It combines the PolyFit reconstruction engine with a feature-preserving mesh workflow, a real-time 3D viewport, optional GPU compute acceleration, PBR texture baking, and export tools for common 3D formats.

## Highlights

- **Two reconstruction workflows:** PolyFit for point clouds and planar structures, plus a Li & Nan-inspired feature-preserving workflow for existing triangle meshes.
- **Automatic engine routing:** AUTO tries PolyFit first and can fall back to the feature-preserving workflow for mesh inputs when the PolyFit worker fails, times out, or exits unexpectedly.
- **Interactive 3D inspection:** review source and reconstructed geometry, surface shading, materials, wireframes, connected faces, and overlays in the viewport.
- **Material-aware asset handling:** import supported meshes with available UVs, textures, vertex colors, normals, and material properties.
- **Texture baking:** create UV atlases and bake available source material data into texture maps.
- **Flexible export:** write GLB, glTF, FBX, OBJ, PLY, STL, 3MF, or OFF files, with material preservation depending on the target format.
- **Optional GPU acceleration:** use supported CUDA, HIP, oneAPI/SYCL, or OpenCL compute paths, with CPU fallback when acceleration is unavailable or unsuitable.
- **Two Windows distribution paths:** install the source project into a private runtime, or build a single-file portable executable.

## Supported data formats

### Input

| Type | Formats | Notes |
|---|---|---|
| Point clouds | `.ply`, `.pcd`, `.xyz`, `.xyzn`, `.xyzrgb`, `.pts`, `.txt`, `.csv`, `.asc` | PCD supports ASCII and uncompressed binary data. `binary_compressed` PCD is not supported. |
| PolyFit data | `.vg`, `.bvg` | `.vg` is read as ASCII point data. `.bvg` is a segmented PolyFit input and is sent directly to PolyFit; raw-point preview is not available for it. |
| Triangle meshes | `.ply`, `.obj`, `.stl`, `.off`, `.glb`, `.gltf`, `.3mf` | Mesh import can retain supported scene transforms and material/vertex attributes. |

### Output

| Format | Typical use | Material / texture behavior |
|---|---|---|
| **GLB** (`.glb`) | Recommended single-file delivery | Supports the broadest PBR export path. Geometry and supported texture resources are packed into one binary file. |
| glTF (`.gltf`) | Editable or inspectable glTF delivery | Stores JSON plus referenced external resources. Supports the broadest PBR export path. |
| FBX (`.fbx`) | DCC and content-creation interchange | Exports geometry, normals, UVs, and Base Color texture. Material values use a Phong-compatible approximation. Texture files are copied beside the FBX. |
| OBJ (`.obj`) | Broad mesh interchange | Exports geometry and Base Color through MTL and image sidecar files. Other PBR maps are not preserved. |
| PLY (`.ply`) | Geometry and vertex-color workflows | Base Color can be represented as vertex colors. Other PBR maps are not preserved. |
| STL (`.stl`) | 3D printing and geometry exchange | Geometry only in this workflow. |
| 3MF (`.3mf`) | 3D printing and manufacturing | Geometry only in this workflow. |
| OFF (`.off`) | Basic polygon mesh exchange | Geometry only. |

> **Format note:** GLB/glTF are the recommended formats when preserving the baked PBR material workflow matters. OBJ, PLY, and FBX have narrower material support; STL, 3MF, and OFF are geometry-focused exports.

## Reconstruction

Polyfitty Studio provides three reconstruction choices:

| Mode | Behavior |
|---|---|
| **AUTO** | Runs PolyFit first. For an existing triangle mesh, a PolyFit error, timeout, or worker-process failure can trigger the Li & Nan feature-preserving workflow on the original input mesh. Point clouds and VG/BVG inputs use PolyFit. |
| **PolyFit** | Uses the PolyFit 1.6.0 engine for point clouds, planar structures, and segmented VG/BVG data. The application prepares and validates the PolyFit input before running its stepwise reconstruction pipeline. |
| **Li & Nan 2021** | Processes an existing triangle mesh with bilateral normal filtering, planar-region extraction, contour-aware refinement, and region-constrained hierarchical QEM simplification. |

The feature-preserving workflow includes application-side safeguards for open boundaries, close overlapping surfaces, and potentially damaging edge collapses. These checks can increase processing time, and the requested target vertex count may not always be reached when preserving topology is more important.

The application can preview intermediate reconstruction stages while processing. Mesh Cleanup can merge coincident vertices on the selected source or result mesh. Processing and cleanup do not overwrite the original source file.

## Viewport and material inspection

The 3D viewport provides several ways to inspect geometry and materials:

- **Render modes:** PBR, Diffuse (Albedo), and Solid Surface.
- **Geometry views:** Wireframe, Connected Faces, and face/edge overlays.
- **Material inspection:** supported base color, vertex color, UV, texture, normal, and material properties from imported assets.
- **Navigation:** orbit, rotate, pan, zoom, and fly-style camera movement.
- **Display aids:** view-aligned grid and dimension/viewport information.

### Default viewport controls

| Action | Control |
|---|---|
| Select a mesh | Left mouse button |
| Orbit around the pivot | `Alt` + left mouse button |
| Rotate the camera | Right mouse button + drag |
| Pan | Middle mouse button + drag |
| Zoom | Mouse wheel |
| Fly movement | Right mouse button + `W` / `A` / `S` / `D` or `Q` / `E` |
| Increase fly speed | `Shift` while flying |
| Frame the available mesh | `F` |
| Toggle Wireframe | `X` |
| Toggle Connected Faces | `C` |
| Toggle the view-aligned grid | `G` |
| Reimport the selected source | `R` |
| Toggle Face Overlay | `Z` |

Camera navigation ends when the initiating mouse button is released or the viewport loses focus.

## Texture baking and PBR export

The baking workflow creates a UV texture atlas for the reconstructed mesh and transfers supported source material information, including texture and vertex-color data where available.

Available bake-map selections include:

- Base Color (Diffuse / Albedo)
- Normal
- Roughness
- Metallic
- Ambient Occlusion
- Height / Displacement
- Opacity / Alpha
- Emissive
- ORM channel packing (Ambient Occlusion / Roughness / Metallic)

The exact maps available for a bake depend on the source asset and selected bake settings. GLB/glTF can preserve the broadest set of supported baked PBR maps. Export controls are adjusted to the capabilities of the selected output format.

## GPU acceleration

Polyfitty Studio detects available compute devices and can use supported backends:

| Hardware / provider | Compute path |
|---|---|
| NVIDIA | CUDA through CuPy |
| AMD | HIP / HIPRTC, including the optional Windows native HIP bridge |
| Intel | oneAPI / SYCL through `dpnp` and `dpctl` |
| Cross-vendor | OpenCL through `pyopencl` |

GPU acceleration is optional and applies to supported compute kernels rather than every step of the application. Device discovery, runtime checks, and kernel validation are performed before a backend is used. When a backend is unavailable, fails validation, or is not suitable for a particular operation, supported work can continue on the CPU. Some operations remain CPU-side by design.

A compatible vendor driver and compute runtime are required for GPU acceleration. Polyfitty Studio does not silently replace Windows display or compute drivers. GPU packages may be downloaded during installation or first-run setup, depending on the selected device and available runtime.

## System requirements

Polyfitty Studio is designed for 64-bit Windows. Reconstruction, mesh inspection, and texture baking can use substantial CPU memory and temporary storage, especially with dense point clouds, high-polygon meshes, or large texture atlases.

The project does not enforce a single hardware threshold, and performance depends heavily on input size, reconstruction mode, texture resolution, and the selected compute backend. The figures below are **practical planning guidelines**, not a formally benchmarked compatibility guarantee.

| Component | Minimum practical baseline | Recommended for regular / larger projects |
|---|---|---|
| Operating system | Windows 10 or Windows 11, 64-bit | Windows 11, 64-bit, fully updated |
| CPU | Modern 64-bit Intel or AMD processor, 4 cores | Recent 6–8 core Intel or AMD processor |
| System memory (RAM) | 16 GB | 32 GB or more; 64 GB can help with very large datasets |
| Graphics for the 3D viewport | Integrated or dedicated GPU with a current Windows graphics driver | Recent dedicated GPU with up-to-date drivers |
| GPU memory (VRAM) | No universal minimum; GPU compute is optional | 8 GB or more for more demanding GPU workloads |
| Free storage | Allow at least 5 GB for installation/runtime files, plus project data | 10–20 GB or more for runtimes, caches, temporary data, and exports |
| Network | Internet access for installation and optional first-run GPU setup | Stable broadband connection for downloads and updates |

### GPU acceleration requirements

A dedicated GPU is **not required** to use the CPU processing path. GPU acceleration is optional and is enabled only when a compatible device, vendor driver, and matching compute runtime are available and pass the application's checks.

Supported compute paths depend on the installed hardware and runtime:

- **NVIDIA:** CUDA 12.x through CuPy. The bundled CUDA 12.6 Update 2 setup checks for a Windows driver compatibility floor of 528.33; driver 560.94 or newer is recommended for the CUDA 12.6 Update 2 driver level.
- **AMD:** HIP / HIPRTC, including the optional Windows native HIP bridge, where the required AMD driver and runtime are available.
- **Intel:** oneAPI / SYCL through `dpnp` and `dpctl` with a compatible Intel GPU/runtime environment.
- **Cross-vendor:** OpenCL through `pyopencl`, provided the device driver exposes a working OpenCL implementation.

The application validates compute devices and can fall back to supported CPU operations when GPU acceleration is unavailable or unsuitable. VRAM needs vary with the operation and dataset; the recommended VRAM figure is guidance, not a hard compatibility limit.

### Runtime and installer notes

The standard installer provisions a private **CPython 3.12.10 x64** runtime and the core Python dependencies. A separate system-wide Python installation is not required. Internet access is needed to download the runtime and dependencies during installation; optional GPU packages may also be downloaded during installation or first-run setup. Keep the project/runtime location writable and allow additional disk space for source assets, processing caches, and exported files.

> **Hardware guidance:** The memory, CPU, VRAM, and storage figures above are estimated starting points, not published minimum specifications verified across every workload. Large point clouds, dense meshes, high-resolution texture baking, and simultaneous applications may require substantially more resources. If a project runs out of memory, reduce input density or texture resolution, close other applications, or use a machine with more RAM/VRAM.

## Installation

### Install from the source package

1. Download or clone this repository, then extract it to a writable folder if needed.
2. Run **`INSTALL.vbs`** or **`INSTALL.bat`** from the project root.
3. Follow the setup manager and allow the runtime, dependencies, and runtime checks to complete.
4. Launch Polyfitty Studio from the created desktop shortcut, **`START_HIDDEN.vbs`**, or **`START.bat`**.

To repair an interrupted or damaged installation, run the installer again. To remove the installed application environment and shortcuts, run **`UNINSTALL.bat`**. Source assets and exported files are kept.

## Build the single-file portable executable

The project includes a Windows x64 builder that packages the application and its core runtime into one executable.

1. Open the complete project source folder on a Windows x64 machine.
2. Run **`BUILD_PORTABLE_EXE.vbs`** or **`BUILD_PORTABLE_EXE.bat`**.
3. Allow the builder to download the required runtime and package prerequisites.
4. Find the generated executable here:

```text
release\portable\PolyfittyStudio.exe
```

The portable build embeds the application source, CPython 3.12.10, Qt, VTK, core scientific libraries, PolyFit, native modules, and required notices. The recipient does not need to extract an application folder or install the Studio runtime.

On first launch, the executable verifies its embedded payload and extracts it to a user-local cache under:

```text
%LOCALAPPDATA%\PF\P\<payload-sha256>
```

Later launches reuse the verified cache. GPU capability checks and optional user-space GPU package setup are performed on the recipient machine; a compatible hardware-vendor driver is still required for GPU acceleration.

To create desktop shortcuts for an already-built portable executable, run **`CREATE_PORTABLE_SHORTCUT.vbs`**.

## Development and tests

The application source is organized into focused modules:

```text
PolyfittyStudio/
├── app/                  # GUI, viewport, I/O, reconstruction, baking, GPU compute
├── installer/            # Install, runtime verification, GPU setup, portable builder
├── native/               # Optional native compute bridge source
├── packaging/            # Startup splash and packaging assets
├── tests/                # Automated test suite
├── requirements.txt      # Core Python dependencies
├── requirements_gpu_*.txt# Optional GPU dependency sets
├── runtime_manifest.json # Versioned source/runtime integrity manifest
├── LICENSE
└── THIRD_PARTY_NOTICES.txt
```

The repository includes a pytest test suite covering reconstruction safety, GPU acceleration and fallback, portable packaging, material/viewport behavior, import/export, and installer workflows.

After installing `pytest` into the project's private runtime, run the tests from the project root:

```powershell
.runtime\python312\python.exe -m pip install pytest
.runtime\python312\python.exe -m pytest -q
```

The test suite is provided with the source; test results depend on the machine and its runtime environment.

## Logs and troubleshooting

- Installer and startup diagnostics are written under the application's **`.cache/logs`** directory.
- Portable startup logs are stored in the extracted Local AppData application cache.
- When startup or GPU setup fails, review the newest log and rerun the relevant setup step.
- If GPU initialization fails, confirm the vendor driver is installed and current. The application can use CPU processing when GPU acceleration is unavailable.
- If a mesh reconstruction is not suitable for PolyFit, use **AUTO** or select **Li & Nan 2021** for an existing triangle mesh.
- For PCD input, use ASCII or uncompressed binary data; compressed binary PCD is not supported.

## Versioning

The source package reports version **0.2.131** in `VERSION.txt` and `runtime_manifest.json`. The runtime manifest also records source-file hashes and feature-version metadata used by the startup/runtime integrity checks.

## License and third-party components

Polyfitty Studio is distributed under the **GNU General Public License v3.0**. See [`LICENSE`](LICENSE) for the full license text.

Third-party dependency and runtime notices are provided in [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt). Review the applicable upstream licenses before redistributing the application, bundled runtimes, or optional GPU components.
