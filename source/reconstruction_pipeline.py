
import correspondence_estimation
import Correspondences
import feature_estimation
import numpy as np
import point_tracking
import RHC
import surface_reconstruction
import torch
import Triangulation


class ReconstructionPipeline:
    def __init__(self, camera, laser, feature_estimator, point_tracker, correspondence_estimator, surface_reconstructor):
        self._camera = camera
        self._laser = laser
        self._feature_estimator: feature_estimation.FeatureEstimator = feature_estimator
        self._point_tracker: point_tracking.PointTracker = point_tracker
        self._correspondence_estimator = correspondence_estimator
        self._surface_reconstructor: surface_reconstruction.SurfaceReconstructor = surface_reconstructor

        # Comes from Feature Extractor
        self._glottal_segmentaitons = None
        self._vocalfold_segmentations = None
        self._glottal_midlines = None
        self._glottal_outlines = None
        self._laserpoint_estimates = None
        self._laserpoint_images = None
        self._glottal_area_waveform = None

        # Comes from point tracker, start and end point are the first and last frame of maximal glottal closure.
        self._optimized_point_positions = None
        self._start_point = None
        self._end_point = None

        # Comes from correspondence matcher
        self._laserpoint_ids = None # Must be same size as self._optimized_point_positions

        # Comes from triangulation
        self._point_clouds = None

        # Comes from surface reconstruction
        self._left_vocalfold_meshes = None
        self._right_vocalfold_meshes = None



    def set_feature_estimator(self, feature_estimator: feature_estimation.FeatureEstimator) -> None:
        self._feature_estimator = feature_estimator

    def set_point_tracker(self, point_tracker: point_tracking.PointTracker) -> None:
        self._point_tracker = point_tracker

    def set_surface_reconstructor(self, surface_reconstructor: surface_reconstruction.SurfaceReconstructor) -> None:
        self._surface_reconstructor = surface_reconstructor

    def set_correspondence_matcher(self, correspondence_matcher: correspondence_estimation.CorrespondenceEstimator) -> None:
        self._correspondence_estimator = correspondence_matcher

    def estimate_features(self, video: torch.tensor) -> None:
        gl_seg, gl_midline, gl_outline, vf_seg, lp_pos, lp_image, gaw = self._feature_estimator.compute_features(video)
        self._glottal_segmentaitons = gl_seg
        self._vocalfold_segmentations = vf_seg
        self._glottal_midlines = gl_midline
        self._glottal_outlines = gl_outline
        self._laserpoint_positions = lp_pos
        self._laserpoint_images = lp_image
        self._glottal_area_waveform = gaw


    def track_points(self, video: torch.tensor) -> torch.tensor:
        self._optimized_point_positions = self._point_tracker.track_points(video, self._feature_estimator)
        return self._optimized_point_positions

    def estimate_correspondences(self, min_depth: float, max_depth: float, consensus_size: int, iterations: int) -> None:
        maximum_closing_frame = self._feature_estimator.glottalAreaWaveform().argmin()
        point_image = self._feature_estimator.create_image_from_points(self._optimized_point_positions[maximum_closing_frame])

        pixelLocations, laserGridIDs = Correspondences.initialize(
                    self._laser,
                    self._camera,
                    point_image.detach().cpu().numpy(),
                    min_depth,
                    max_depth,
                )
        
        self.grid2DPixLocations = RHC.RHC(
            laserGridIDs,
            pixelLocations,
            point_image.detach().cpu().numpy(),
            self._camera,
            self._laser,
            consensus_size,
            iterations,
        )


        # Given that we now have a list of laser beam IDS and pixel positions,
        # we now need to find the corresponding points in self._optimized_point_positions
        # as we can not guarantee, that the ordering stayed coherent.
        # So, we iterate over 

        placeholder_correspondences: np.array = np.zeros(self._optimized_point_positions[maximum_closing_frame].shape, dtype=int) * np.nan
        np_point_positions: np.array = self._optimized_point_positions.detach().cpu().numpy()[maximum_closing_frame]

        for laserpoint_id, pixel_position in self.grid2DPixLocations:
            corresponding_id = (np_point_positions.astype(int) == pixel_position.astype(int)).all(axis=-1).argmax()
            placeholder_correspondences[corresponding_id] = laserpoint_id

        self.point_correspondences = placeholder_correspondences
        mask = ~np.isnan(self.point_correspondences).any(axis=-1)
        self.filtered_point_correspondences = self.point_correspondences[mask].astype(int)
        self.filtered_optimized_points = self._optimized_point_positions.detach().cpu().numpy()[:, mask, :]


        #self._laserpoint_ids = self._correspondence_estimator.estimate_correspondences(self.camera, self.laser, self._optimized_point_positions, maximum_closing_frame)

    def triangulation(self, min_interval, max_interval) -> None:
        self._point_clouds = Triangulation.triangulationMatNew(
            self._camera, 
            self._laser, 
            self.filtered_point_correspondences, 
            self.filtered_optimized_points, 
            min_interval, 
            max_interval, 
            min_interval, 
            max_interval)
        return self._point_clouds

    def surface_reconstruction(self) -> None:
        self._surface_reconstructor.compute_surface(self._feature_estimator, self._point_clouds)

    def reconstruct(self):
        self.estimate_features()
        self.track_points()
        self.estimate_correspondences()
        self.triangulation()
        self.surface_reconstruction()