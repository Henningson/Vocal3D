
import correspondence_estimation
import feature_estimation
import point_tracking
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

    def estimate_features(self) -> None:
        gl_seg, gl_midline, gl_outline, vf_seg, lp_pos, lp_image, gaw = self._feature_estimator.compute_features()
        self._glottal_segmentaitons = gl_seg
        self._vocalfold_segmentations = vf_seg
        self._glottal_midlines = gl_midline
        self._glottal_outlines = gl_outline
        self._laserpoint_positions = lp_pos
        self._laserpoint_images = lp_image
        self._glottal_area_waveform = gaw


    def track_points(self, video: torch.tensor) -> torch.tensor:
        self._optimized_point_positions = self._point_tracker.track_points(video, self._feature_estimator)
        point_video = []
        for frame, points in zip(video, self._optimized_point_positions):
            image = self._point_tracker.draw_points_on_image(frame, points)
            point_video.append(image.detach().cpu().numpy())
        
        return point_video

    def estimate_correspondences(self) -> None:
        _, maximum_closing_frame = self._glottal_area_waveform.max()
        self._laserpoint_ids = self._correspondence_estimator.estimate_correspondences(self.camera, self.laser, self._optimized_point_positions, maximum_closing_frame)

    def triangulation(self, min_interval, max_interval) -> None:
        self._point_clouds = Triangulation.triangulationMat(self._camera, self._laser, self._correspondence_estimator.correspondences(), min_interval, max_interval, min_interval, max_interval)

    def surface_reconstruction(self) -> None:
        self._surface_reconstructor.compute_surface(self._feature_estimator, self._point_clouds)

    def reconstruct(self):
        self.estimate_features()
        self.track_points()
        self.estimate_correspondences()
        self.triangulation()
        self.surface_reconstruction()