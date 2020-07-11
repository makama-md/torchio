import torch
from torchio import Subject, Image
from torchio.transforms import RandomNoise
from ..utils import TorchioTestCase


class TestReproducibility(TorchioTestCase):

    def setUp(self):
        super().setUp()
        self.subject = Subject(img=Image(tensor=torch.ones(4, 4, 4)))

    def random_stuff(self, seed=None):
        transform = RandomNoise(std=(100, 100), seed=seed)
        transformed = transform(self.subject)
        value = transformed.img.data.sum().item()
        random_params_dict = transformed.history[0][1]
        return value, random_params_dict['seed']

    def test_reproducibility_no_seed(self):
        a, seed_a = self.random_stuff()
        b, seed_b = self.random_stuff()
        self.assertNotEqual(a, b)
        c, seed_c = self.random_stuff(seed_a)
        self.assertEqual(c, a)
        self.assertEqual(seed_c, seed_a)

    def test_reproducibility_seed(self):
        torch.manual_seed(42)
        a, seed_a = self.random_stuff()
        b, seed_b = self.random_stuff()
        self.assertNotEqual(a, b)
        c, seed_c = self.random_stuff(seed_a)
        self.assertEqual(c, a)
        self.assertEqual(seed_c, seed_a)

        torch.manual_seed(42)
        a2, seed_a2 = self.random_stuff()
        self.assertEqual(a2, a)
        self.assertEqual(seed_a2, seed_a)
        b2, seed_b2 = self.random_stuff()
        self.assertNotEqual(a2, b2)
        self.assertEqual(b2, b)
        self.assertEqual(seed_b2, seed_b)
        c2, seed_c2 = self.random_stuff(seed_a2)
        self.assertEqual(c2, a2)
        self.assertEqual(seed_c2, seed_a2)
        self.assertEqual(c2, c)
        self.assertEqual(seed_c2, seed_c)
