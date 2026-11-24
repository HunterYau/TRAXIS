import unittest
import numpy as np
from examples.pzt_era_stable import split


def response(model, count=64):
    x=model['b'][:,0].copy();out=[]
    for _ in range(count):
        out.append(model['c']@x);x=model['a']@x
    return np.asarray(out)


class StableSeparationTests(unittest.TestCase):
    def test_coupled_schur_blocks_preserve_additive_response(self):
        model=dict(a=np.array([[.9,.2],[0.,1.02]]),b=np.array([[1.],[.03]]),c=np.array([[1.,.5],[-.4,.7]]))
        stable,unstable,info=split(model)
        np.testing.assert_allclose(response(stable)+response(unstable),response(model),atol=1e-13)
        np.testing.assert_allclose(stable['b'],[[.95]],atol=1e-13)
        self.assertEqual(info['retained_stable_states'],1)
        self.assertFalse(info['poles_moved'])

    def test_complex_conjugate_pair_stays_together(self):
        a=np.zeros((4,4));a[:2,:2]=[[.8,-.2],[.2,.8]];a[2:,2:]=[[1.,-.3],[.3,1.]]
        a[:2,2:]=[[.1,.2],[.3,-.2]]
        model=dict(a=a,b=np.arange(1.,5.)[:,None],c=np.array([[1.,-.3,.2,.4]]))
        stable,unstable,info=split(model)
        self.assertEqual(info['retained_stable_states'],2)
        np.testing.assert_allclose(response(stable)+response(unstable),response(model),rtol=1e-12,atol=1e-12)
        self.assertLess(max(abs(np.linalg.eigvals(stable['a']))),1.)

    def test_all_stable_or_all_unstable(self):
        for eigen in (.9,1.1):
            model=dict(a=np.array([[eigen]]),b=np.ones((1,1)),c=np.ones((2,1)))
            stable,unstable,_=split(model)
            np.testing.assert_allclose(response(stable)+response(unstable),response(model))


if __name__=='__main__':unittest.main()
