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
