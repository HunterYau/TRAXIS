"""Check the independent displacement recurrence against an analytic ODE."""
import math
import unittest

import numpy as np
from scipy import sparse

from examples.embedded_native_pulse import central_difference
from traxis.resources import ResourceLimitError


class NativePulseTests(unittest.TestCase):
    def test_forced_mode_final_displacement_velocity_have_second_order_error(self):
        omega,drive,duration=3.,1.,2.
        exact_u=(math.sin(drive*duration)-drive/omega*math.sin(omega*duration))/(omega**2-drive**2)
        exact_v=drive*(math.cos(drive*duration)-math.cos(omega*duration))/(omega**2-drive**2)
        errors=[]
        for steps in (32,64,128):
            trace,u,v=central_difference(sparse.csr_matrix([[omega**2]]),np.ones(1),np.ones(1),np.ones((1,1)),
                steps,duration/steps,lambda t:math.sin(drive*t))
            self.assertEqual(trace.shape,(steps+1,1)); self.assertEqual(trace[0,0],0.)
            self.assertEqual(trace[-1,0],u[0]); errors.append([abs(u[0]-exact_u),abs(v[0]-exact_v)])
        ratios=np.array(errors[:-1])/np.array(errors[1:])
        self.assertTrue(np.all((ratios>3.8)&(ratios<4.2)))

    def test_guard_failure_propagates(self):
        class Guard:
            def check(self): raise ResourceLimitError('system RAM limit reached')
        with self.assertRaises(ResourceLimitError):
            central_difference(sparse.eye(1),np.ones(1),np.ones(1),np.ones((1,1)),32,.01,lambda _:0.,Guard())


if __name__=='__main__': unittest.main()
