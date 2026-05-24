"""Rendering stats for diverse point-maze envs.

Only the ``RENDER_STATS`` dict is consumed by stable-worldmodel: the drawer,
``PixelMapper``, and camera-config logic read ``obs_min_total`` /
``obs_range_total`` / ``lookat`` / ``image_width`` per env root.

Training-schedule constants (episode lengths, sampling modes, etc.) live in
Hydra configs under ``scripts/data/config/`` and are intentionally not
duplicated here.
"""

RENDER_STATS = {
    'maze2d-umaze-v1': {
        'lookat': [3, 3, 0],
        'image_topleft_in_obs_coord': [-2.1729, 5.7517],
        'scale_coord_obs_to_pixel': 63,
        'arrow_mult': 30,
    },
    'maze2d-medium-v1': {
        'lookat': [4.5, 4.5, 0],
        'image_topleft_in_obs_coord': [-1.95, 8.55],
        'scale_coord_obs_to_pixel': (500 / 10.5),
        'arrow_mult': 25,
        'transformed': True,
        'crop_left_top': [100, 100],
        'scale_factor': 300 / 64,
    },
    'maze2d_large_diverse': {
        'lookat': [5.5, 5.5, 0],
        'image_topleft_in_obs_coord': [-1.95, 8.55],
        'scale_coord_obs_to_pixel': (500 / 10.5),
        'arrow_mult': 25,
        'transformed': True,
        'crop_left_top': [10, 10],
        'scale_factor': 386 / 98,
        'image_width': 98,
        'obs_min_space': 0.351,
        'obs_max_space': 8.248,
        'obs_range_total': 10.068675,
        'obs_min_total': -0.636125,
    },
    'maze2d_medium_diverse': {
        'lookat': [4.5, 4.5, 0],
        'image_topleft_in_obs_coord': [-1.95, 8.55],
        'scale_coord_obs_to_pixel': (500 / 10.5),
        'arrow_mult': 25,
        'transformed': True,
        'crop_left_top': [10, 10],
        'scale_factor': 386 / 81,
        'image_width': 81,
        'obs_min_space': 0.353,
        'obs_max_space': 6.246,
        'obs_range_total': 7.857,
        'obs_min_total': -0.629,
    },
    'maze2d_small_diverse': {
        'lookat': [3.5, 3.5, 0],
        'image_topleft_in_obs_coord': [-1.95, 8.55],
        'scale_coord_obs_to_pixel': (500 / 10.5),
        'arrow_mult': 25,
        'transformed': True,
        'crop_left_top': [77, 77],
        'scale_factor': 345 / 64,
        'image_width': 64,
        'obs_min_space': 0.35,
        'obs_max_space': 4.2485,
        'obs_range_total': 5.84775,
        'obs_min_total': -0.6246,
    },
    'maze2d-large-v1': {
        'lookat': [5.0, 6.5, 0],
        'image_topleft_in_obs_coord': [-3.06, 12.15],
        'scale_coord_obs_to_pixel': 36.5,
        'arrow_mult': 20,
    },
}
