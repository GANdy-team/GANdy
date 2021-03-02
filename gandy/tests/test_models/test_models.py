"""Testing functions for UncertaintyModel parent class"""
import unittest
import gandy.models.models


class TestUncertaintyModel(unittest.TestCase):
    
    def test___init__(self):
        """Test initialization of the UncertaintyModel class"""
        ## first mock the build method
        with subject as gandy.models.models.UncertaintyModel:
            mocked_build = unittest.mock.MagicMock() 
            subject.build = mocked_build
            ## initialize
            self.subject = subject(xshape=(6,),yshape=(3,))
            ## test assignment of shapes
            self.assertTrue(hasattr(self.subject, 'xshape'))
            self.assertTrue(hasattr(self.subject, 'yshape'))
            ## test that build was called
            self.subject.build.assert_called()
            ## test that we initializzed sessions
            self.assertEqual(self.subject.sessions, [])
        return        