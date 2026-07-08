"""Texture-transfer utilities for qualitative correspondence figures.

Follows the standard protocol (cf. ULRSSM's utils/texture_util.py): synthesize
per-vertex UV coordinates on the source shape X from its two highest-variance
axes, then carry them to the target Y through the predicted map. Two transfer
variants are produced side by side:

  * hard:     uv_y = uv_x[p2p]                      (the map the metrics score)
  * smoothed: uv_y = Phi_y @ Cxy @ Phi_x^+ @ uv_x   (spectrally low-passed)

The hard transfer is the honest view of the evaluated p2p map; the smoothed one
matches how published figures (e.g. ULRSSM) are rendered, but hides
high-frequency error — always read the two together.

Rendering uses a small numpy software rasterizer (orthographic projection,
z-buffer, per-pixel barycentric UV interpolation) so the repo's ``texture.png``
(a labelled 8x8 color grid, as in ULRSSM's figures) is sampled at full image
resolution — per-vertex color sampling cannot resolve the grid labels. Figures
are composed headlessly with matplotlib.
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')  # headless: safe on a remote machine with no display
import matplotlib.pyplot as plt

from paths import REPO_ROOT

# the labelled color-grid texture used for all transfer figures
DEFAULT_TEXTURE_FILE = os.path.join(REPO_ROOT, 'texture.png')


# --------------------------------------------------------------------------- #
# UV synthesis + transfer
# --------------------------------------------------------------------------- #
def generate_tex_coords(verts):
    """Synthesize per-vertex UVs by projecting onto the two highest-variance axes
    (ULRSSM's generate_tex_coords), min-max normalized to [0, 1]^2.

    Args:
        verts (np.ndarray): vertex positions, [V, 3].
    Returns:
        uv (np.ndarray): texture coordinates, [V, 2].
    """
    ind = np.argsort(np.std(verts, axis=0))[::-1]
    vt = np.stack([verts[:, ind[1]], verts[:, ind[0]]], axis=-1)
    vt = vt - vt.min(axis=0, keepdims=True)
    vt = vt / (vt.max(axis=0, keepdims=True) + 1e-12)
    return vt


def hard_uv_transfer(uv_x, p2p_yx):
    """Transfer UVs through the raw point-to-point map: uv_y[i] = uv_x[p2p[i]].

    Args:
        uv_x (np.ndarray): source UVs, [Vx, 2].
        p2p_yx (np.ndarray): point map Y -> X (index into X per Y vertex), [Vy].
    Returns:
        uv_y (np.ndarray): target UVs, [Vy, 2].
    """
    return uv_x[p2p_yx]


def smoothed_uv_transfer(uv_x, p2p_yx, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y):
    """Transfer UVs through the spectrally smoothed map Pyx = Phi_y Cxy Phi_x^+.

    Cxy is estimated from the p2p map (Cxy = Phi_y^+ Phi_x[p2p]), then Pyx is
    applied without ever materializing the dense [Vy, Vx] matrix. Truncating to
    the eigenbasis low-passes the map, so this view smooths over local error —
    pair it with ``hard_uv_transfer`` for an honest comparison.

    Args:
        uv_x (np.ndarray): source UVs, [Vx, 2].
        p2p_yx (np.ndarray): point map Y -> X, [Vy].
        evecs_x (np.ndarray): LB eigenvectors of X, [Vx, K].
        evecs_y (np.ndarray): LB eigenvectors of Y, [Vy, K].
        evecs_trans_x (np.ndarray): mass-weighted transposed eigenvectors of X
            (Phi_x^+ = Phi_x^T M_x), [K, Vx].
        evecs_trans_y (np.ndarray): same for Y, [K, Vy].
    Returns:
        uv_y (np.ndarray): target UVs, [Vy, 2], clipped back to [0, 1].
    """
    Cxy = evecs_trans_y @ evecs_x[p2p_yx]                # [K, K]
    uv_y = evecs_y @ (Cxy @ (evecs_trans_x @ uv_x))      # Pyx @ uv_x, factored
    return np.clip(uv_y, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# textures
# --------------------------------------------------------------------------- #
def load_texture(texture_file=DEFAULT_TEXTURE_FILE):
    """Load the texture image as float RGB in [0, 1]; falls back to a procedural
    checkerboard when the file is unavailable."""
    if not os.path.isfile(texture_file):
        return checkerboard_image()
    img = plt.imread(texture_file)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(img[..., :3])  # drop alpha if present


def checkerboard_image(res=512, n_cells=6):
    """Procedural fallback texture: a UV-driven color ramp modulated by a
    light/dark checker so texture direction stays readable.

    Args:
        res (int): image resolution (square). Default 512.
        n_cells (int): checker cells per axis. Default 6.
    Returns:
        texture (np.ndarray): RGB image in [0, 1], [res, res, 3].
    """
    v, u = np.meshgrid(np.linspace(1, 0, res), np.linspace(0, 1, res), indexing='ij')
    cell = (np.floor(u * n_cells) + np.floor(v * n_cells)).astype(np.int64)
    base = np.stack([0.35 + 0.65 * u, 0.45 + 0.55 * v, 1.0 - 0.55 * u], axis=-1)
    shade = np.where(cell % 2 == 0, 1.0, 0.5)[..., None]
    return np.clip(base * shade, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# correspondence vertex colors
# --------------------------------------------------------------------------- #
def create_colormap(verts):
    """Per-vertex RGB from the min-max-normalized vertex positions (ULRSSM's
    create_colormap): a smooth position-coded rainbow, so after transferring
    through the map, matching regions on the two shapes share the same color.

    Args:
        verts (np.ndarray): vertex positions, [V, 3].
    Returns:
        colors (np.ndarray): RGB in [0, 1], [V, 3].
    """
    lo, hi = verts.min(axis=0, keepdims=True), verts.max(axis=0, keepdims=True)
    return (verts - lo) / (hi - lo + 1e-12)


# --------------------------------------------------------------------------- #
# software rasterizer
# --------------------------------------------------------------------------- #
def _view_rotation(elev_deg=15.0, azim_deg=-60.0):
    """Orthographic camera basis (right, up, forward) from elevation/azimuth."""
    e, a = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
    # direction from the object center toward the camera
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    forward = -d                                     # camera looks along -d
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)    # rows: screen x, y, depth


def _mesh_to_zup(verts, flip_up=False):
    """Map mesh coordinates to a z-up world frame, preserving handedness.

    Datasets here are y-up (``flip_up=False``) or negative-y-up (SMAL,
    ``flip_up=True``).
    """
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    if flip_up:
        return np.stack([x, -z, -y], axis=-1)
    return np.stack([x, z, y], axis=-1)


def rasterize_mesh_attribute(verts, faces, attr, attr_to_rgb, res=700,
                             flip_up=False, elev_deg=15.0, azim_deg=-60.0):
    """Render a mesh with per-pixel interpolation of a per-vertex attribute
    (orthographic, z-buffered).

    The attribute is interpolated with barycentric coordinates inside every
    triangle and mapped to RGB per pixel via ``attr_to_rgb`` — e.g. a texture
    lookup for UVs, or the identity for vertex colors. Faces are lambert-shaded
    from their normals so the geometry stays readable.

    Args:
        verts (np.ndarray): vertex positions, [V, 3].
        faces (np.ndarray): triangle indices, [F, 3].
        attr (np.ndarray): per-vertex attribute, [V, K].
        attr_to_rgb (callable): maps interpolated attributes [P, K] -> RGB [P, 3].
        res (int): output image resolution (square). Default 700.
        flip_up (bool): dataset up-axis flag (True for SMAL).
        elev_deg, azim_deg (float): orthographic view angles.
    Returns:
        image (np.ndarray): RGBA image in [0, 1], [res, res, 4] (alpha 0 = empty).
    """
    v = _mesh_to_zup(np.asarray(verts, dtype=np.float64), flip_up)
    v = v @ _view_rotation(elev_deg, azim_deg).T     # columns: screen x, y, depth

    # fit the mesh into the image with a small margin (equal aspect)
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    scale = (res - 1) / (2 * half)
    px = (v[:, 0] - center[0] + half) * scale        # screen x -> pixel col
    py = (v[:, 1] - center[1] + half) * scale        # screen y -> pixel row (flipped later)
    depth = v[:, 2]

    tri = np.stack([px, py, depth], axis=-1)[faces]      # [F, 3, 3]
    tri_attr = np.asarray(attr, dtype=np.float64)[faces] # [F, 3, K]

    # flat lambert shading from world-space face normals
    w = _mesh_to_zup(np.asarray(verts, dtype=np.float64), flip_up)[faces]
    n = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)
    light = np.array([0.3, -0.5, 0.8])
    light = light / np.linalg.norm(light)
    face_shade = 0.45 + 0.55 * np.abs(n @ light)     # [F]

    color = np.ones((res, res, 3), dtype=np.float64)
    alpha = np.zeros((res, res), dtype=np.float64)
    zbuf = np.full((res, res), np.inf, dtype=np.float64)

    # painter-free z-buffer rasterization, one triangle at a time
    for f in range(tri.shape[0]):
        t = tri[f]
        x0, y0 = int(max(0, np.floor(t[:, 0].min()))), int(max(0, np.floor(t[:, 1].min())))
        x1 = int(min(res - 1, np.ceil(t[:, 0].max())))
        y1 = int(min(res - 1, np.ceil(t[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue

        # barycentric coordinates on the pixel grid of the bounding box
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = ((t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0])
             + (t[2, 0] - t[1, 0]) * (t[0, 1] - t[2, 1]))
        if abs(d) < 1e-12:
            continue
        b0 = ((t[1, 1] - t[2, 1]) * (gx - t[2, 0]) + (t[2, 0] - t[1, 0]) * (gy - t[2, 1])) / d
        b1 = ((t[2, 1] - t[0, 1]) * (gx - t[2, 0]) + (t[0, 0] - t[2, 0]) * (gy - t[2, 1])) / d
        b2 = 1.0 - b0 - b1
        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not inside.any():
            continue

        z = b0 * t[0, 2] + b1 * t[1, 2] + b2 * t[2, 2]
        rows, cols = gy.astype(np.int64), gx.astype(np.int64)
        visible = inside & (z < zbuf[rows, cols])
        if not visible.any():
            continue
        rs, cs = rows[visible], cols[visible]
        zbuf[rs, cs] = z[visible]

        # per-pixel attribute interpolation + RGB mapping
        vals = (b0[visible, None] * tri_attr[f, 0]
                + b1[visible, None] * tri_attr[f, 1]
                + b2[visible, None] * tri_attr[f, 2])    # [P, K]
        color[rs, cs] = np.clip(attr_to_rgb(vals) * face_shade[f], 0.0, 1.0)
        alpha[rs, cs] = 1.0

    image = np.concatenate([color, alpha[..., None]], axis=-1)
    return image[::-1]                               # pixel row 0 = image top


def rasterize_textured_mesh(verts, faces, uv, texture, **kwargs):
    """Render a mesh with per-pixel UV texture lookup — this is what makes the
    labelled texture grid legible (per-vertex sampling cannot resolve it).
    See ``rasterize_mesh_attribute`` for the shared arguments.

    Args:
        uv (np.ndarray): per-vertex texture coordinates in [0, 1]^2, [V, 2].
        texture (np.ndarray): RGB image in [0, 1], [H, W, 3].
    """
    th, tw = texture.shape[:2]

    def lookup(uvs):
        tc = np.minimum((np.clip(uvs[:, 0], 0, 1) * tw).astype(np.int64), tw - 1)
        tr = np.minimum(((1.0 - np.clip(uvs[:, 1], 0, 1)) * th).astype(np.int64), th - 1)
        return texture[tr, tc]

    return rasterize_mesh_attribute(verts, faces, uv, lookup, **kwargs)


def rasterize_colored_mesh(verts, faces, colors, **kwargs):
    """Render a mesh with smoothly interpolated per-vertex RGB colors.
    See ``rasterize_mesh_attribute`` for the shared arguments.

    Args:
        colors (np.ndarray): per-vertex RGB in [0, 1], [V, 3].
    """
    return rasterize_mesh_attribute(verts, faces, colors,
                                    lambda c: np.clip(c, 0.0, 1.0), **kwargs)


# --------------------------------------------------------------------------- #
# figure composition
# --------------------------------------------------------------------------- #
def render_texture_transfer_figure(verts_x, faces_x, verts_y, faces_y, p2p_yx,
                                   evecs_x=None, evecs_y=None,
                                   evecs_trans_x=None, evecs_trans_y=None,
                                   flip_up=False, res=700, title=None,
                                   texture_file=DEFAULT_TEXTURE_FILE,
                                   out_file=None):
    """Render a source/hard/smoothed texture-transfer comparison figure.

    The smoothed panel is only drawn when all four spectral arrays are given
    (otherwise the figure has two panels).

    Args:
        verts_x, faces_x (np.ndarray): source mesh X, [Vx, 3] / [Fx, 3].
        verts_y, faces_y (np.ndarray): target mesh Y, [Vy, 3] / [Fy, 3].
        p2p_yx (np.ndarray): predicted point map Y -> X, [Vy].
        evecs_* / evecs_trans_* (np.ndarray, optional): spectral quantities for
            the smoothed transfer (see ``smoothed_uv_transfer``).
        flip_up (bool): dataset up-axis flag (True for SMAL).
        res (int): per-panel raster resolution. Default 700.
        title (str, optional): figure suptitle.
        texture_file (str): texture image sampled at the UVs (default: the
            repo's texture.png); falls back to a procedural checkerboard when
            the file is missing.
        out_file (str, optional): path to save the PNG; the figure is closed
            after saving. When omitted the open figure is returned.
    Returns:
        fig (matplotlib.figure.Figure or None): None when saved to ``out_file``.
    """
    texture = load_texture(texture_file)
    uv_x = generate_tex_coords(verts_x)
    rasterize = lambda verts, faces, uv: rasterize_textured_mesh(
        verts, faces, uv, texture, res=res, flip_up=flip_up)
    return _compose_transfer_figure(
        uv_x, rasterize, verts_x, faces_x, verts_y, faces_y, p2p_yx,
        (evecs_x, evecs_y, evecs_trans_x, evecs_trans_y), title, out_file)


def render_color_transfer_figure(verts_x, faces_x, verts_y, faces_y, p2p_yx,
                                 evecs_x=None, evecs_y=None,
                                 evecs_trans_x=None, evecs_trans_y=None,
                                 flip_up=False, res=700, title=None,
                                 out_file=None):
    """Render a source/hard/smoothed vertex-color correspondence figure.

    The source is colored by its normalized positions (``create_colormap``, as
    in ULRSSM's qualitative point-cloud figures) and the colors are carried to
    the target through the map — matching regions share the same color. Same
    panel layout and arguments as ``render_texture_transfer_figure`` (minus the
    texture file).
    """
    colors_x = create_colormap(verts_x)
    rasterize = lambda verts, faces, colors: rasterize_colored_mesh(
        verts, faces, colors, res=res, flip_up=flip_up)
    return _compose_transfer_figure(
        colors_x, rasterize, verts_x, faces_x, verts_y, faces_y, p2p_yx,
        (evecs_x, evecs_y, evecs_trans_x, evecs_trans_y), title, out_file)


def _compose_transfer_figure(signal_x, rasterize, verts_x, faces_x,
                             verts_y, faces_y, p2p_yx, spectral, title, out_file):
    """Shared panel layout: source / hard p2p / (optional) smoothed spectral.

    ``signal_x`` is any per-vertex signal on X (UVs or RGB); ``rasterize`` maps
    (verts, faces, signal) to an image. The hard/smoothed transfers are the
    same linear maps regardless of what the signal encodes.
    """
    panels = [('source X', verts_x, faces_x, signal_x),
              ('target Y — hard p2p', verts_y, faces_y,
               hard_uv_transfer(signal_x, p2p_yx))]
    if all(s is not None for s in spectral):
        panels.append(('target Y — smoothed (spectral)', verts_y, faces_y,
                       smoothed_uv_transfer(signal_x, p2p_yx, *spectral)))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (name, verts, faces, signal) in zip(axes, panels):
        ax.imshow(rasterize(verts, faces, signal))
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if out_file is not None:
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig
