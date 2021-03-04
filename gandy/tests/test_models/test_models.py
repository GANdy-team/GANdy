"""Testing functions for UncertaintyModel parent class"""
import numpy
import unittest

import gandy.models.models as mds


class TestUncertaintyModel(unittest.TestCase):
    
    def test___init__(self):
        """Test initialization of the UncertaintyModel class"""
        ## first mock the build method
        with unittest.patch(
            'gandy.models.models.UncertaintyModel.build'
        ) as mocked_build:
            # initialize
            subject = mds.UncertaintyModel(xshape=(6,),
                                           yshape=(3,),
                                           keyword=5)  # keywords passed?
            # test assignment of shapes
            self.assertTrue(hasattr(subject, 'xshape'))
            self.assertTrue(hasattr(subject, 'yshape'))
            # test that build was called
            mocked_build.assert_called_once_with(keyword=5)
            # test that we initializzed sessions
            self.assertEqual(subject.sessions, [])
        return
    
    def test_check(self):
        """Test the ability of the model to recognize improper data"""
        # prepare some data objects to check.
        # we only have numpy available in the dependencies
        # should work with other objects such as pandas dataframe
        # test different dimensions
        XSHAPE = [(5,6,), (8,)]; YSHAPE = [(5,), (1,)]
        for xshape, yshape in XSHAPE, YSHAPE:
            Xs_good = numpy.ones(
                (20, *xshape), # as if it were 20 data points
                dtype=int      # specify int to ensure proper conversion to float
            )
            Xs_bad = numpy.ones(
                (20, 3, 4)
            )
            Xs_non_numeric = XS_GOOD.astype('str')
            Ys_good = numpy.ones(
                (20, *yshape) # matching 20 data points
            )
            Ys_bad = numpy.ones(
                (20, 3)
            )
            Ys_datacount_mismatch = numpy.ones(
                (10, *yshape) # not matching 20 data points
            )
            no_shape_attribute = [1,2,3]

            # prepare the subject
            subject = mds.UncertaintyModel(xshape, yshape)
            # first test only Xs passed
            # start with expected success
            Xs_out = subject.check(Xs_good)
            self.assertEqual(numpy.ndarray, type(Xs_out))
            self.assertEqual(Xs_good.shape, Xs_out.shape)
            self.assertEqual(numpy.float64, Xs_out.dtype)
            # failure modes
            with self.assertRaises(ValueError):
                subject.check(Xs_bad)
            with self.assertRaises(TypeError):
                subject.check(Xs_non_numeric)
            with self.assertRaises(AttributeError):
                subject.check(no_shape_attribute)
            # Xs and y together
            # expected success
            Xs_out, Ys_out = subject.check(Xs_good, Ys_good)
            self.assertTrue(numpy.ndarray == type(Xs_out) and \
                            numpy.ndarray == type(Ys_out))
            self.assertTrue(Xs_good.shape == Xs_out.shape and \
                            Ys_good.shape == Ys_out.shape)
            self.assertEqual(numpy.float64, Xs_out.dtype)
            # failure modes
            with self.assertRaises(ValueError):
                subject.check(Xs_bad, Ys_good)
            with self.assertRaises(ValueError):
                subject.check(Xs_good, Ys_bad)
            with self.assertRaises(TypeError):
                subject.check(Xs_non_numeric, Ys_good)
            with self.assertRaises(AttributeError):
                subject.check(no_shape_attribute, Ys_good)
            with self.assertRaises(AttributeError):
                subject.check(Xs_good, no_shape_attribute)
            with self.assertRaises(ValueError):
                subject.check(Xs_good, Ys_datacount_mismatch)
        return
        
                         