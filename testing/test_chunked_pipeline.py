import unittest
import torch
from torch.utils.data import Dataset

# Adjust import path as necessary. Assuming the file is in the project root.
# If 'misc' is not directly importable, sys.path manipulation might be needed for standalone execution,
# but for structured projects, this should work if tests are run from the project root.
try:
    from misc.datasets.datasets import LabelChunkedTaskDataset
except ImportError:
    # Fallback for local testing if path issues occur, e.g. if run from 'testing' directory
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from misc.datasets.datasets import LabelChunkedTaskDataset


class MockOriginalDataset(Dataset):
    def __init__(self, data_samples): # data_samples = [(image_tensor, label_int), ...]
        self.samples = data_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Returns image, original_label
        return self.samples[idx]


class TestLabelChunkedTaskDataset(unittest.TestCase):
    def setUp(self):
        # Create mock data
        self.mock_data_tuples = [
            (torch.randn(3, 32, 32), 0),    # Index 0
            (torch.randn(3, 32, 32), 25),   # Index 1
            (torch.randn(3, 32, 32), 49),   # Index 2
            (torch.randn(3, 32, 32), 50),   # Index 3
            (torch.randn(3, 32, 32), 75),   # Index 4
            (torch.randn(3, 32, 32), 99),   # Index 5
            (torch.randn(3, 32, 32), 150),  # Index 6 (for default task)
        ]
        self.original_dataset = MockOriginalDataset(self.mock_data_tuples)
        
        self.task_configs = [
            {'task_id': 0, 'original_labels': {'range': [0, 50]}, 'num_classes': 50}, # Actual labels 0-49
            {'task_id': 1, 'original_labels': {'range': [50, 100]}, 'num_classes': 50}# Actual labels 50-99
        ]
        
        self.default_task_id = -1
        self.default_task_specific_label = -100 # A distinct label for default task
        
        self.chunked_dataset = LabelChunkedTaskDataset(
            original_dataset=self.original_dataset,
            task_configs=self.task_configs,
            default_task_id=self.default_task_id,
            default_task_specific_label=self.default_task_specific_label
        )

    def test_instantiation_and_length(self):
        self.assertEqual(len(self.chunked_dataset), len(self.original_dataset))

    def _get_sample_by_original_label(self, original_label):
        idx = next(i for i, sample_tuple in enumerate(self.mock_data_tuples) if sample_tuple[1] == original_label)
        # __getitem__ returns: image, (task_specific_label, task_detector_target), original_label, assigned_task_id_val
        return self.chunked_dataset[idx], idx

    def test_sample_chunk0_label0(self):
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(0)
        self.assertIsNotNone(img)
        self.assertIsInstance(targets, tuple)
        self.assertEqual(len(targets), 2)
        self.assertEqual(orig_label_out, 0)
        self.assertEqual(task_id_val, 0)
        self.assertEqual(targets[0].item(), 0) # task_specific_label (0 remapped from 0)
        self.assertTrue(torch.equal(targets[1], torch.tensor([1.0, 0.0])))

    def test_sample_chunk0_label25(self):
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(25)
        self.assertIsNotNone(img)
        self.assertEqual(orig_label_out, 25)
        self.assertEqual(task_id_val, 0)
        self.assertEqual(targets[0].item(), 25) # task_specific_label (25 remapped from 25)
        self.assertTrue(torch.equal(targets[1], torch.tensor([1.0, 0.0])))

    def test_sample_chunk0_label49(self):
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(49)
        self.assertIsNotNone(img)
        self.assertEqual(orig_label_out, 49)
        self.assertEqual(task_id_val, 0)
        self.assertEqual(targets[0].item(), 49) # task_specific_label (49 remapped from 49)
        self.assertTrue(torch.equal(targets[1], torch.tensor([1.0, 0.0])))

    def test_sample_chunk1_label50(self):
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(50)
        self.assertIsNotNone(img)
        self.assertEqual(orig_label_out, 50)
        self.assertEqual(task_id_val, 1)
        # Remapped: 50 (original) - 50 (start of chunk range) = 0
        self.assertEqual(targets[0].item(), 0) # task_specific_label 
        self.assertTrue(torch.equal(targets[1], torch.tensor([0.0, 1.0])))

    def test_sample_chunk1_label75(self):
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(75)
        self.assertIsNotNone(img)
        self.assertEqual(orig_label_out, 75)
        self.assertEqual(task_id_val, 1)
        # Remapped: 75 (original) - 50 (start of chunk range) = 25
        self.assertEqual(targets[0].item(), 25) # task_specific_label
        self.assertTrue(torch.equal(targets[1], torch.tensor([0.0, 1.0])))

    def test_sample_chunk1_label99(self):
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(99)
        self.assertIsNotNone(img)
        self.assertEqual(orig_label_out, 99)
        self.assertEqual(task_id_val, 1)
        # Remapped: 99 (original) - 50 (start of chunk range) = 49
        self.assertEqual(targets[0].item(), 49) # task_specific_label
        self.assertTrue(torch.equal(targets[1], torch.tensor([0.0, 1.0])))

    def test_get_task_info(self):
        task_info = self.chunked_dataset.get_task_info()
        self.assertEqual(task_info['num_distinct_task_chunks_for_predictor'], 2)
        
        details = task_info['main_task_chunks_details']
        self.assertEqual(len(details), 2)
        
        self.assertEqual(details[0]['task_id'], 0)
        self.assertEqual(details[0]['num_classes'], 50)
        self.assertEqual(details[0]['original_label_range'], [0, 49]) # Note: range in config is [0,50)
        
        self.assertEqual(details[1]['task_id'], 1)
        self.assertEqual(details[1]['num_classes'], 50)
        self.assertEqual(details[1]['original_label_range'], [50, 99]) # Note: range in config is [50,100)

    def test_default_task_id_for_out_of_range_label(self):
        # Original label 150 is out of defined task ranges
        (img, targets, orig_label_out, task_id_val), _ = self._get_sample_by_original_label(150)
        self.assertIsNotNone(img)
        self.assertEqual(orig_label_out, 150)
        self.assertEqual(task_id_val, self.default_task_id) # Should be -1
        self.assertEqual(targets[0].item(), self.default_task_specific_label) # Should be -100
        
        # task_detector_target should indicate no specific task chunk
        # For num_distinct_tasks = 2, this means [0.0, 0.0]
        self.assertTrue(torch.equal(targets[1], torch.tensor([0.0, 0.0])))

if __name__ == '__main__':
    # This allows running the test script directly
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# --- New Test Classes Start Here ---
import torch.nn as nn
from unittest.mock import patch

try:
    from model.modules.backbones import MultiTailResNetBackbone
    from model.network import FiLMedNetworkWithSharedStem
    # For mocking factories if FiLMedNetworkWithSharedStem uses them for internal instantiation
    import model.modules.stem 
    import model.modules.task_predictors
    import model.modules.module_registry
    import model.modules.backbones # Though we use the real one, get_model might be called
    import sc2bench.models.layer # For get_layer
except ImportError:
    # Adjusting sys.path if running tests from a different directory or if project structure is complex
    import sys
    import os
    if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from model.modules.backbones import MultiTailResNetBackbone
    from model.network import FiLMedNetworkWithSharedStem
    import model.modules.stem 
    import model.modules.task_predictors
    import model.modules.module_registry
    import model.modules.backbones
    import sc2bench.models.layer


class TestMultiTailResNetBackbone(unittest.TestCase):
    def setUp(self):
        self.input_features_dim = 64
        self.num_tails = 2
        self.per_tail_output_classes = 50
        self.batch_size = 4
        self.spatial_dim = 8 # Example spatial dimension from g_s

        self.backbone = MultiTailResNetBackbone(
            input_features_dim=self.input_features_dim,
            num_tails=self.num_tails,
            per_tail_output_classes=self.per_tail_output_classes
        )

    def test_output_structure_and_shape(self):
        x_features = torch.randn(self.batch_size, self.input_features_dim, self.spatial_dim, self.spatial_dim)
        # Conditioning signal is not strictly used by this backbone's forward logic for selection,
        # but passed for compatibility with the network structure.
        conditioning_signal = torch.randn(self.batch_size, self.num_tails) 
        
        output_dict = self.backbone(x_features, conditioning_signal)

        self.assertIsInstance(output_dict, dict)
        self.assertIn('tail_0', output_dict)
        self.assertIn('tail_1', output_dict)

        self.assertIsInstance(output_dict['tail_0'], torch.Tensor)
        self.assertEqual(output_dict['tail_0'].shape, (self.batch_size, self.per_tail_output_classes))

        self.assertIsInstance(output_dict['tail_1'], torch.Tensor)
        self.assertEqual(output_dict['tail_1'].shape, (self.batch_size, self.per_tail_output_classes))


# Mock component classes for TestFiLMedNetworkWithSharedStemOutput
class MockSharedStem(nn.Module):
    def __init__(self, **kwargs): # Accept dummy params
        super().__init__()
        self.output_channels = 64
        self.conv = nn.Conv2d(3, self.output_channels, kernel_size=3, padding=1) # Dummy layer

    def forward(self, x): # x: (B, 3, H, W)
        # Simulate some processing, e.g. changing spatial dimensions
        return self.conv(torch.randn(x.size(0), 3, x.size(2)*2, x.size(3)*2, device=x.device)) # (B, 64, H_out, W_out)
    
    def get_output_channels(self):
        return self.output_channels

class MockTaskProbabilityModel(nn.Module):
    def __init__(self, **kwargs): # Accept dummy params like input_channels_from_stem, num_tasks
        super().__init__()
        self.num_tasks = kwargs.get('num_tasks', 2)
        # Dummy layer, e.g. a linear layer after flattening and pooling
        input_feat = kwargs.get('input_channels_from_stem', 64) * 16 * 16 # Example, ensure it aligns
        self.fc = nn.Linear(input_feat, self.num_tasks)


    def forward(self, x_stem_features): # x_stem_features: (B, C_stem, H_stem, W_stem)
        batch_size = x_stem_features.size(0)
        # Flatten and pass through a dummy layer
        # x_stem_features = x_stem_features.mean(dim=[2,3]) # Global avg pool
        # return self.fc(x_stem_features)
        return torch.randn(batch_size, self.num_tasks, device=x_stem_features.device) # (B, num_tasks)

    def get_output_dim(self):
        return self.num_tasks

class MockCompressionModule(nn.Module):
    def __init__(self, **kwargs): # Accept dummy params like analysis_config, etc.
        super().__init__()
        # Mocking g_s and its final_layer for the FiLMedNetworkWithSharedStem __init__
        self.g_s = nn.Sequential(nn.Identity()) 
        self.g_s.final_layer = nn.Identity() # Critical for FiLMedNetwork's init logic for reconstruction_for_backbone
        self.output_channels = 64 # Example, should match backbone input_features_dim

    def forward(self, x_stem_features, conditioning_signal): # (B, C_stem, H_stem, W_stem), (B, num_tasks)
        batch_size = x_stem_features.size(0)
        # Simulate output that goes to the backbone
        return torch.randn(batch_size, self.output_channels, 8, 8, device=x_stem_features.device) # (B, C_backbone_in, H_backbone_in, W_backbone_in)

    def update(self, force=False): # Add dummy update method
        return True
        
    def compress(self, obj, **kwargs): return obj # Dummy
    def decompress(self, obj, **kwargs): return obj # Dummy


class TestFiLMedNetworkWithSharedStemOutput(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_tails = 2
        self.per_tail_output_classes = 50
        self.stem_output_channels = 64
        self.backbone_input_features_dim = 64 # from MockCompressionModule output

        # Configurations matching the mocks and real backbone
        self.shared_stem_config = {'params': {'input_channels': 3, 'output_channels': self.stem_output_channels}} # Dummy params for mock
        self.task_probability_model_config = {
            'params': {'input_channels_from_stem': self.stem_output_channels, 'num_tasks': self.num_tails}
        }
        # analysis_config needs film_cond_dim matching task_prob_model output_dim
        self.compression_module_config = {
            'name': 'MockCompressionModule', # Will be mocked by get_custom_compression_module
            'params': {
                'analysis_config': {'params': {'input_channels_from_stem': self.stem_output_channels, 'film_cond_dim': self.num_tails}},
                'synthesis_config': {'params': {}}, # Dummy
                'hyper_analysis_config': {'params': {}}, # Dummy
                'hyper_synthesis_config': {'params': {}} # Dummy
            }
        }
        self.backbone_config = {
            'name': 'MultiTailResNetBackbone', # Will be mocked by get_model to return real one
            'params': {
                'input_features_dim': self.backbone_input_features_dim,
                'num_tails': self.num_tails,
                'per_tail_output_classes': self.per_tail_output_classes
            }
        }
        self.reconstruction_layer_for_backbone_config = {'name': 'Identity', 'params': {}}
        self.analysis_config_parent = {}

        # Patch the factory functions/classes used in FiLMedNetworkWithSharedStem.__init__
        # The 'target' strings must match where these are looked up by FiLMedNetworkWithSharedStem
        
        self.patch_shared_stem = patch('model.network.SharedInputStem', # Assuming it's imported as 'SharedInputStem' in model.network
                                       return_value=MockSharedStem(**self.shared_stem_config['params']))
        self.patch_task_model = patch('model.network.TaskProbabilityModel', 
                                      return_value=MockTaskProbabilityModel(**self.task_probability_model_config['params']))
        self.patch_get_compressor = patch('model.network.get_custom_compression_module', 
                                          return_value=MockCompressionModule(**self.compression_module_config['params']))
        
        # For the backbone, we want to use the *real* MultiTailResNetBackbone
        self.real_backbone_instance = MultiTailResNetBackbone(**self.backbone_config['params'])
        self.patch_get_backbone = patch('model.network.get_model', 
                                        return_value=self.real_backbone_instance)
        
        # For reconstruction_for_backbone, if 'Identity' is used, get_layer should return nn.Identity
        self.patch_get_layer = patch('sc2bench.models.layer.get_layer', return_value=nn.Identity())

        # Start the patches
        self.mock_stem_factory = self.patch_shared_stem.start()
        self.mock_task_model_factory = self.patch_task_model.start()
        self.mock_compressor_factory = self.patch_get_compressor.start()
        self.mock_backbone_factory = self.patch_get_backbone.start()
        self.mock_layer_factory = self.patch_get_layer.start()

        self.filmed_network = FiLMedNetworkWithSharedStem(
            shared_stem_config=self.shared_stem_config,
            task_probability_model_config=self.task_probability_model_config,
            compression_module_config=self.compression_module_config,
            backbone_config=self.backbone_config,
            reconstruction_layer_for_backbone_config=self.reconstruction_layer_for_backbone_config,
            analysis_config_parent=self.analysis_config_parent
        )

    def tearDown(self):
        self.patch_shared_stem.stop()
        self.patch_task_model.stop()
        self.patch_get_compressor.stop()
        self.patch_get_backbone.stop()
        self.patch_get_layer.stop()

    def test_forward_pass_output_structure(self):
        dummy_image = torch.randn(self.batch_size, 3, 32, 32)
        output = self.filmed_network(dummy_image)

        self.assertIsInstance(output, dict)
        self.assertIn('backbone_outputs', output)
        self.assertIn('conditioning_signal', output)

        backbone_outputs = output['backbone_outputs']
        self.assertIsInstance(backbone_outputs, dict)
        self.assertIn('tail_0', backbone_outputs)
        self.assertIn('tail_1', backbone_outputs)

        self.assertEqual(backbone_outputs['tail_0'].shape, (self.batch_size, self.per_tail_output_classes))
        self.assertEqual(backbone_outputs['tail_1'].shape, (self.batch_size, self.per_tail_output_classes))

        self.assertEqual(output['conditioning_signal'].shape, (self.batch_size, self.num_tails))


# --- Tests for Loss and Metrics ---
try:
    from misc.loss import ChunkedCrossEntropyLoss
    from misc.eval import TaskPredictionAccuracy, AccuracyForChunk
except ImportError:
    # sys.path already manipulated above, so this should ideally not be needed again
    # but kept for robustness if this section is run in isolation somehow.
    import sys
    import os
    if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from misc.loss import ChunkedCrossEntropyLoss
    from misc.eval import TaskPredictionAccuracy, AccuracyForChunk

class TestChunkedCrossEntropyLoss(unittest.TestCase):
    def setUp(self):
        self.num_chunks = 2
        self.num_classes_per_chunk = 3 # Smaller for easier manual logit creation
        self.criterion = ChunkedCrossEntropyLoss(
            num_chunks=self.num_chunks,
            task_loss_weight=0.5,
            chunk_loss_weight=0.5,
            label_smoothing=0.0
        )

    def test_loss_calculation_perfect_prediction(self):
        batch_size = 2
        # Sample 0 -> task 0, label 1 (remapped)
        # Sample 1 -> task 1, label 2 (remapped)
        model_output = {
            "conditioning_signal": torch.tensor([[10.0, -10.0], [-10.0, 10.0]]), # Perfect task prediction
            "backbone_outputs": {
                "tail_0": torch.nn.functional.one_hot(torch.tensor([1, 0]), num_classes=self.num_classes_per_chunk).float() * 10.0, 
                          # For sample 0 (task 0): predicts label 1. Sample 1 output for tail_0 is ignored.
                "tail_1": torch.nn.functional.one_hot(torch.tensor([0, 2]), num_classes=self.num_classes_per_chunk).float() * 10.0
                          # For sample 1 (task 1): predicts label 2. Sample 0 output for tail_1 is ignored.
            }
        }
        # Ensure batch_size consistency for tail outputs
        model_output["backbone_outputs"]["tail_0"][1,:] = -10.0 # Irrelevant part for sample 1 / tail 0
        model_output["backbone_outputs"]["tail_1"][0,:] = -10.0 # Irrelevant part for sample 0 / tail 1


        targets = (
            torch.tensor([1, 2]), # task_specific_label
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]) # task_detector_target (one-hot)
        )
        loss = self.criterion(model_output, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=4) # Softmax CE loss won't be exactly 0 with finite logits

    def test_loss_calculation_completely_wrong_prediction(self):
        batch_size = 2
        # Sample 0 -> task 0, label 1. Model predicts task 1, label 0.
        # Sample 1 -> task 1, label 2. Model predicts task 0, label 1.
        model_output = {
            "conditioning_signal": torch.tensor([[-10.0, 10.0], [10.0, -10.0]]), # Wrong task prediction
            "backbone_outputs": {
                "tail_0": torch.nn.functional.one_hot(torch.tensor([0, 1]), num_classes=self.num_classes_per_chunk).float() * 10.0,
                "tail_1": torch.nn.functional.one_hot(torch.tensor([1, 0]), num_classes=self.num_classes_per_chunk).float() * 10.0,
            }
        }
        targets = (
            torch.tensor([1, 2]), # task_specific_label
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]) # task_detector_target (one-hot)
        )
        loss = self.criterion(model_output, targets)
        # Check that loss is substantially higher than near-zero
        # Exact value depends on CE, but should be positive and significant
        self.assertTrue(loss.item() > 1.0) 

    def test_loss_with_mixed_batch_and_weights(self):
        batch_size = 4
        model_output = {
            "conditioning_signal": torch.tensor([ # Logits for tasks
                [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0] 
            ]), 
            "backbone_outputs": {
                "tail_0": torch.randn(batch_size, self.num_classes_per_chunk),
                "tail_1": torch.randn(batch_size, self.num_classes_per_chunk)
            }
        }
        targets = (
            torch.tensor([0, 1, 2, 0]), # task_specific_labels (remapped)
            torch.tensor([[1.0,0.0],[0.0,1.0],[1.0,0.0],[0.0,1.0]]) # task_detector_target
        )
        
        self.criterion.task_loss_weight = 0.5
        self.criterion.chunk_loss_weight = 0.5
        loss1 = self.criterion(model_output, targets)
        self.assertIsInstance(loss1.item(), float)

        self.criterion.task_loss_weight = 1.0 # Emphasize task loss
        self.criterion.chunk_loss_weight = 0.1
        loss2 = self.criterion(model_output, targets)
        self.assertIsInstance(loss2.item(), float)
        
        # Test only task loss (chunk loss weight = 0)
        self.criterion.task_loss_weight = 1.0
        self.criterion.chunk_loss_weight = 0.0
        loss_task_only = self.criterion(model_output, targets)
        
        # Test only chunk loss (task loss weight = 0)
        self.criterion.task_loss_weight = 0.0
        self.criterion.chunk_loss_weight = 1.0
        loss_chunk_only = self.criterion(model_output, targets)

        # Combined loss should be sum if weights are 1 and 1 (after adjusting criterion)
        self.criterion.task_loss_weight = 1.0
        self.criterion.chunk_loss_weight = 1.0
        loss_sum_check = self.criterion(model_output, targets)
        self.assertAlmostEqual(loss_sum_check.item(), (loss_task_only + loss_chunk_only).item(), places=5)


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.task_acc_metric = TaskPredictionAccuracy()
        self.chunk0_acc_metric = AccuracyForChunk(chunk_id=0)
        self.chunk1_acc_metric = AccuracyForChunk(chunk_id=1)
        self.num_classes_per_chunk = 3 # For dummy logits

    def test_task_prediction_accuracy(self):
        self.task_acc_metric.reset()
        model_output = {
            "conditioning_signal": torch.tensor([
                [10.0, -10.0], # Correctly predicts task 0
                [-10.0, 10.0], # Correctly predicts task 1
                [9.0, -9.0]    # Correctly predicts task 0 (was intended to be wrong, fixed for clarity)
            ]),
            "backbone_outputs": {} # Not needed for this metric
        }
        targets = (
            torch.empty(3), # Not needed
            torch.tensor([[1.0,0.0],[0.0,1.0],[1.0,0.0]]) # True tasks: 0, 1, 0
        )
        self.task_acc_metric.update(model_output, targets)
        self.assertAlmostEqual(self.task_acc_metric.compute(), 3.0/3.0)

        self.task_acc_metric.reset()
        model_output_v2 = {
            "conditioning_signal": torch.tensor([
                [10.0, -10.0], # Correct task 0
                [10.0, -10.0], # Incorrect, predicts task 0 but is task 1
                [-10.0, 9.0]   # Correct task 1
            ]),
             "backbone_outputs": {}
        }
        targets_v2 = (
            torch.empty(3),
            torch.tensor([[1.0,0.0],[0.0,1.0],[0.0,1.0]]) # True tasks: 0, 1, 1
        )
        self.task_acc_metric.update(model_output_v2, targets_v2)
        self.assertAlmostEqual(self.task_acc_metric.compute(), 2.0/3.0)


    def test_accuracy_for_chunk0(self):
        self.chunk0_acc_metric.reset()
        # 3 samples, all belong to chunk 0. Tail 0 predicts 2 correctly.
        # Predictions for tail_0: label 1 (correct), label 0 (incorrect), label 2 (correct)
        # True labels for chunk 0: label 1, label 1, label 2
        model_output = {
            "conditioning_signal": torch.tensor([[10.,-10.],[10.,-10.],[10.,-10.]]), # All predicted as task 0
            "backbone_outputs": {
                "tail_0": torch.tensor([ [0,10,0], [10,0,0], [0,0,10] ], dtype=torch.float), # Preds: 1, 0, 2
                "tail_1": torch.randn(3, self.num_classes_per_chunk) # Dummy for other tail
            }
        }
        targets = (
            torch.tensor([1, 1, 2]), # True task_specific_labels for these chunk 0 samples
            torch.tensor([[1.,0.],[1.,0.],[1.,0.]]) # All actually task 0
        )
        self.chunk0_acc_metric.update(model_output, targets)
        self.assertAlmostEqual(self.chunk0_acc_metric.compute(), 2.0/3.0)

    def test_accuracy_for_chunk_no_samples(self):
        self.chunk0_acc_metric.reset()
        # All samples belong to chunk 1
        model_output = {
            "conditioning_signal": torch.tensor([[-10.,10.],[-10.,10.]]), # All predicted as task 1
            "backbone_outputs": {
                "tail_0": torch.randn(2, self.num_classes_per_chunk), 
                "tail_1": torch.tensor([ [10,0,0], [0,10,0] ], dtype=torch.float) # Preds for chunk 1
            }
        }
        targets = (
            torch.tensor([0, 1]), # True task_specific_labels for chunk 1 samples
            torch.tensor([[0.,1.],[0.,1.]]) # All actually task 1
        )
        self.chunk0_acc_metric.update(model_output, targets)
        self.assertAlmostEqual(self.chunk0_acc_metric.compute(), 0.0)

    def test_accuracy_for_chunk_mixed_batch(self):
        self.chunk0_acc_metric.reset()
        self.chunk1_acc_metric.reset()
        
        # Sample 0: chunk 0, label 1, predicted label 1 (correct)
        # Sample 1: chunk 1, label 0, predicted label 0 (correct)
        # Sample 2: chunk 0, label 2, predicted label 0 (incorrect)
        # Sample 3: chunk 1, label 1, predicted label 2 (incorrect)
        model_output = {
            "conditioning_signal": torch.tensor([ # Perfect task prediction
                [10.,-10.], [-10.,10.], [10.,-10.], [-10.,10.]
            ]),
            "backbone_outputs": {
                "tail_0": torch.tensor([ # For chunk 0 samples (idx 0, 2)
                    [0,10,0], # Pred for sample 0: label 1
                    [-1,0,0], # Dummy for sample 1 (not chunk 0)
                    [10,0,0], # Pred for sample 2: label 0
                    [-1,0,0]  # Dummy for sample 3 (not chunk 0)
                ], dtype=torch.float),
                "tail_1": torch.tensor([ # For chunk 1 samples (idx 1, 3)
                    [-1,0,0], # Dummy for sample 0 (not chunk 1)
                    [10,0,0], # Pred for sample 1: label 0
                    [-1,0,0], # Dummy for sample 2 (not chunk 1)
                    [0,0,10]  # Pred for sample 3: label 2
                ], dtype=torch.float)
            }
        }
        targets = (
            torch.tensor([1, 0, 2, 1]), # task_specific_labels (remapped)
            torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]) # task_detector_target (true tasks: 0, 1, 0, 1)
        )

        # Test chunk 0 accuracy (samples 0 and 2)
        # Sample 0: true label 1, predicted 1 (correct)
        # Sample 2: true label 2, predicted 0 (incorrect)
        # Expected acc for chunk 0: 1/2 = 0.5
        self.chunk0_acc_metric.update(model_output, targets)
        self.assertAlmostEqual(self.chunk0_acc_metric.compute(), 1.0/2.0)

        # Test chunk 1 accuracy (samples 1 and 3)
        # Sample 1: true label 0, predicted 0 (correct)
        # Sample 3: true label 1, predicted 2 (incorrect)
        # Expected acc for chunk 1: 1/2 = 0.5
        self.chunk1_acc_metric.update(model_output, targets)
        self.assertAlmostEqual(self.chunk1_acc_metric.compute(), 1.0/2.0)
