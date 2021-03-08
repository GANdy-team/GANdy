"""Testing functions for UncertaintyModel parent class"""
import numpy
import unittest
import unittest.mock

import gandy.models.models as mds
import gandy.quality_est.metrics


class TestUncertaintyModel(unittest.TestCase):

    @unittest.mock.patch('gandy.models.models.UncertaintyModel.build')
    def test___init__(self, mocked_build):
        """Test initialization of the UncertaintyModel class"""
        # first mock the build method
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
        self.assertEqual(subject.sessions, {})
        self.assertTrue(hasattr(subject, 'model'))
        return

    @unittest.mock.patch('gandy.models.models.UncertaintyModel.build')
    def test_check(self, mocked_build):
        """Test the ability of the model to recognize improper data"""
        # prepare some data objects to check.
        # we only have numpy available in the dependencies
        # should work with other objects such as pandas dataframe
        # test different dimensions
        XSHAPE = [(5, 6,), (8,)]
        YSHAPE = [(5,), (1,)]
        for xshape, yshape in XSHAPE, YSHAPE:
            Xs_good = numpy.ones(
                (20, *xshape),  # as if it were 20 data points
                dtype=int      # specify int to ensure proper conversion
            )
            Xs_bad = numpy.ones(
                (20, 3, 4)
            )
            Xs_non_numeric = numpy.array(['str'])
            Ys_good = numpy.ones(
                (20, *yshape)  # matching 20 data points
            )
            Ys_bad = numpy.ones(
                (20, 3)
            )
            Ys_datacount_mismatch = numpy.ones(
                (10, *yshape)  # not matching 20 data points
            )
            no_shape_attribute = [1, 2, 3]

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
            self.assertTrue(isinstance(Xs_out, numpy.ndarray) and
                            isinstance(Ys_out, numpy.ndarray))
            self.assertTrue(Xs_good.shape == Xs_out.shape and
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

    def test_build(self):
        """Test the parent build method, to make sure it executes protected
        method"""
        model = 'Mymodel'
        with unittest.mock.patch(
            'gandy.models.models.UncertaintyModel._build',
            return_value=model  # mock the return of the model to a string
        ) as mocked__build:
            subject = mds.UncertaintyModel((1,), (1,), keyword=5)
            mocked__build.assert_called_once_with(keyword=5)
            # ensure automatically set model
            self.assertTrue(subject.model is model)
        return

    def test__build(self):
        """Parent _build should do nothing but raise"""
        with self.assertRaises(mds.NotImplimented):
            mds.UncertaintyModel((1,), (1,))
        # mock _build from here on out - we don;t want the init build to
        # interfere
        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test__get_metric(self, mocked__build):
        """test ability to retrieve the correct callables"""
        def fake_metric(trues, predictions, uncertainties):
            return 5
        # initialize the subject
        subject = mds.UncertaintyModel((1,), (1,))
        # try all success cases
        metric_out = subject._get_metric(fake_metric)
        self.assertEqual(fake_metric, metric_out)
        metric_out = subject._get_metric('Metric')
        self.assertEqual(gandy.quality_est.metrics.Metric, metric_out)
        metric_out = subject._get_metric(None)
        self.assertTrue(metric_out is None)
        # and failure, not a class
        with self.assertRaises(AttributeError):
            subject._get_metric('not_a_class')

        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test_train(self, mocked__build):
        """Proper passing of data to _train and updating of sessions"""
        subject = mds.UncertaintyModel((1,), (1,))
        # mock the required nested calls
        Xs_in, Ys_in = 'Xs', 'Ys'
        mocked_check = unittest.mock.MagicMock(
            return_value=('Xs_checked', 'Ys_checked')
        )
        subject.check = mocked_check
        mocked__train = unittest.mock.MagicMock(
            return_value='losses'
        )
        subject._train = mocked__train
        mocked__get_metric = unittest.mock.MagicMock(
            return_value='some_metric'
        )
        subject._get_metric = mocked__get_metric
        # run the train and check proper calls
        with unittest.mock.patch('time.clock', return_value='thetime'
                                 ) as mocked_time:
            # first specify a session name
            subject.train(Xs_in, Ys_in,
                          metric='fake_metric',
                          session='first_session')
            mocked_check.assert_called_with(Xs_in, Ys_in)
            mocked__get_metric.assert_called_with('fake_metric')
            mocked__train.assert_called_with('Xs_checked',
                                             'Ys_checked',
                                             metric='some_metric')
            # then try without specifying session name, we want to make its own
            # also don't give a metric to make sure that is an allowed option
            subject.train(Xs_in, Ys_in)
            mocked_time.assert_called()
            # check all of the correct storing of sessions
            self.assertEqual(2, len(subject.sessions))
            self.assertTrue('first_session' in subject.sessions.keys())
            self.assertEqual(subject.sessions['first_session'], 'losses')
            self.assertTrue('Starttime: thetime' in subject.sessions.keys())
        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test__train(self, mocked__build):
        """All it should do is raise an error for child to define"""
        subject = mds.UncertaintyModel((1,), (1,))
        with self.assertRaises(mds.NotImplimented):
            subject._train('Xs', 'Ys')
        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test_predict(self, mocked__build):
        """Test proper flagging of predictions"""
        subject = mds.UncertaintyModel((1,), (1,))
        # prepare and mock objects
        Xs_in = 'Xs'
        # here we set up a rotation of predictions, uncertaintains for
        # _predict to return, allowing us to test _predict output handling
        _predict_return = [
            ([5, 10], numpy.array([5, 10], dtype=int)),  # works
            (['length', '2'], ['wrong', 'dtype']),  # failure, can't flag
            (['length', '2'], 5.0),  # failure, pred/unc length mismatch
            ('length1', 5.0),  # failure, does not match length of input
            ([['array', 'is'], ['2', 'dim']], [5, 10])  # yshape mismatch
        ]
        # mock the check and _predict method
        mocked_check = unittest.mock.MagicMock(
            return_value=('Xs_checked')
        )
        subject.check = mocked_check
        mocked__predict = unittest.mock.MagicMock(
            side_effect=_predict_return
        )
        subject._predict = mocked__predict
        # expected faulure, threshold not correct type
        with self.assertRaises(TypeError):
            subject.predict(Xs_in, uc_threshold='not_number')
        # first rotation, expected to work, check outputs and correct calls
        preds, uncs, flags = subject.predict(Xs_in,
                                             uc_threshold=7.0,
                                             keyword=5)

        subject.check.assert_called_with(Xs_in)
        subject._predict.assert_called_with('Xs_checked', keyword=5)

        self.assertTrue(all([isinstance(out, numpy.ndarray) for out in
                             [preds, uncs, flags]]))
        self.assertEqual((2, *subject.yshape), preds.shape)
        self.assertEqual((2, 1), uncs.shape)
        self.assertEqual(uncs.dtype, numpy.float64)
        self.assertEqual((2, 1), flags.shape)
        self.assertEqual(flags.dtype, bool)
        self.assertTrue(numpy.array_equal(flags,
                                          numpy.array([[False], [True]])))
        # first failure case, can't flag strings
        with self.assertRaises(TypeError):
            subject.predict(Xs_in)
        # lengths of pred/unc do not match
        with self.assertRaises(ValueError):
            subject.predict(Xs_in)
        # not the same length as input
        with self.assertRaises(ValueError):
            subject.predict(Xs_in)
        # wrong dimensions
        with self.assertRaises(ValueError):
            subject.predict(Xs_in)

        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test__predict(self, mocked__build):
        """Should just raise an error"""
        subject = mds.UncertaintyModel((1,), (1,))
        with self.assertRaises(mds.NotImplimented):
            subject._predict('Xs')
        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test_score(self, mocked__build):
        """Test proper handling of internal function when score is called"""
        subject = mds.UncertaintyModel((1,), (1,))
        Xs = 'Xs'
        Ys = 'Ys'
        # mock necessary inner calls
        mocked_check = unittest.mock.MagicMock(
            return_value=('Xs_checked', 'Ys_checked')
        )
        subject.check = mocked_check
        mocked_predict = unittest.mock.MagicMock(
            return_value=('preds', 'uncertainties')
        )
        subject.predict = mocked_predict

        def fake_metric1(true, preds, uncertainties):
            return true + preds + uncertainties, [1, 1]

        def fake_metric2(true, preds, uncertainties):
            return true + preds + uncertainties, [1, 1, 1]
        mocked__get_metric = unittest.mock.MagicMock(
            side_effect=[fake_metric1, fake_metric2]
        )
        subject._get_metric = mocked__get_metric
        # excute first run, expected success, metric returns correct len
        value, values = subject.score(Xs, Ys, metric='some_metric', keyword=5)
        mocked__get_metric.assert_called_with('some_metric')
        mocked_check.assert_called_with(Xs, Ys)
        mocked_predict.assert_called_with('Xs_checked', keyword=5)
        self.assertEqual(value, 'Ys_checkedpredsuncertainties')
        self.assertEqual(numpy.ndarray, type(values))
        self.assertEqual((2, 1), values.shape)
        # check that it can find failures in metric computation
        with self.assertRaises(ValueError):
            subject.score(Xs, Ys, metric='some_metric')
        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test_property_shapes(self, mocked__build):
        """Ensure that nonsensical shapes cannot be set"""
        subject = mds.UncertaintyModel((1,), (1,))
        bad_shapes_to_test = ['not tuple', ('tuple', 'of', 'not', 'int')]
        for bad_shape in bad_shapes_to_test:
            with self.assertRaises(TypeError):
                subject.xshape = bad_shape
            with self.assertRaises(TypeError):
                subject.yshape = bad_shape
        # make sure that these setters delete the model as well
        subject._model = 'Not None'
        subject.xshape = (5,)
        self.assertEqual(subject.model, None)
        subject._model = 'Not None'
        subject.yshape = (5,)
        self.assertEqual(subject.model, None)
        return

    @unittest.mock.patch(
        'gandy.models.models.UncertaintyModel._build',
        return_value='Model'
    )
    def test_property_model(self, mocked__build):
        """ensure safety of the model attribute"""
        subject = mds.UncertaintyModel((1,), (1,))
        with self.assertRaises(RuntimeError):
            subject.model = 'Not None'
        return
