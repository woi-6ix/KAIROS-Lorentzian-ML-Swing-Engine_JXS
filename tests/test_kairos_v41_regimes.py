"""Numeric mirror and source invariants; no TradingView compile or profit claims."""
import itertools
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unittest
from hmm_reference import emissions, expectation, fit, forward, gate, Stream

ROOT = Path(__file__).resolve().parents[1]


class HMMTests(unittest.TestCase):
    def assert_probability(self, values):
        self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in values))
        self.assertAlmostEqual(sum(values), 1, places=10)

    def test_forward_matches_explicit_two_step_path_sum(self):
        means, variances = [-0.7, 0, 0.6], [0.2, 0.05, 0.3]
        transitions = [[.9,.08,.02],[.1,.8,.1],[.03,.07,.9]]
        previous = [.2,.5,.3]
        x = 0.4
        e = emissions(x, means, variances)
        joint = [[previous[i]*transitions[i][j]*e[j] for j in range(3)] for i in range(3)]
        total = sum(map(sum, joint))
        expected = [sum(joint[i][j] for i in range(3))/total for j in range(3)]
        for a,b in zip(forward(x, previous, means, variances, transitions), expected):
            self.assertAlmostEqual(a,b,places=12)

    def test_scaled_baum_welch_matches_independent_brute_force_paths(self):
        xs, start = [-.4, .1, .8, .2], [.3,.4,.3]
        means, variances = [-.7,0,.6], [.2,.1,.3]
        trans = [[.9,.08,.02],[.1,.8,.1],[.03,.07,.9]]
        paths = []
        for states in itertools.product(range(3), repeat=len(xs)):
            p = start[states[0]]
            for t,j in enumerate(states):
                # Raw Gaussian densities, independent of scaled emissions.
                p *= math.exp(-.5*(xs[t]-means[j])**2/variances[j])/math.sqrt(2*math.pi*variances[j])
                if t: p *= trans[states[t-1]][j]
            paths.append((states,p))
        total = sum(p for _,p in paths)
        occ, first, second = [0.0]*3, [0.0]*3, [0.0]*3
        counts, initial = [[0.0]*3 for _ in range(3)], [0.0]*3
        for states,p in paths:
            p /= total
            initial[states[0]] += p
            for t,j in enumerate(states):
                occ[j] += p
                first[j] += p*xs[t]
                second[j] += p*xs[t]**2
                if t: counts[states[t-1]][j] += p
        actual = expectation(xs,start,means,variances,trans)
        for a,b in zip(actual, (occ, first, second, counts, initial)):
            aa = list(itertools.chain.from_iterable(a)) if isinstance(a[0],list) else a
            bb = list(itertools.chain.from_iterable(b)) if isinstance(b[0],list) else b
            for x,y in zip(aa,bb): self.assertAlmostEqual(x,y,places=10)

    def test_fit_separated_synthetic_regimes_and_stochastic_transitions(self):
        rng = random.Random(18)
        xs = [rng.gauss(mu,.12) for mu in [-.8,0,.8,-.8,0,.8] for _ in range(50)]
        means, variances, trans, tail = fit(xs,iterations=5)
        self.assertLess(means[0],-.6)
        self.assertLess(abs(means[1]),.1)
        self.assertGreater(means[2],.6)
        self.assertTrue(all(v >= .01 for v in variances))
        self.assert_probability(tail)
        for i,row in enumerate(trans):
            self.assert_probability(row)
            self.assertGreater(row[i],.8)
        for x,state in [(-.8,0),(0,1),(.8,2)]:
            p = [1/3]*3
            for _ in range(10): p=forward(x,p,means,variances,trans)
            self.assertEqual(max(range(3),key=p.__getitem__),state)

    def test_flat_extreme_and_constant_input_is_finite(self):
        for xs in ([0.0]*100,[6.0]*100,[-6.0]*100,[6,-6]*50,[1e-12]*100):
            means, variances, trans, p = fit(xs)
            self.assertEqual(means,sorted(means))
            self.assertTrue(all(math.isfinite(v) for v in means+variances))
            self.assert_probability(p)
            for row in trans: self.assert_probability(row)
            for x in [-6,0,6]: self.assert_probability(forward(x,p,means,variances,trans))
        means,*_ = fit([0.0]*100)
        self.assertFalse(gate(1,True,True,2,2,.99,means))
        self.assertFalse(gate(-1,True,True,0,2,.99,means))

    def test_all_new_toggles_off_exactly_bypass_environment_gates(self):
        for baseline,lc,vol,adx,hmm in itertools.product([False,True],repeat=5):
            # With each new switch OFF, their observed result cannot matter.
            result = baseline and (not False or lc) and (not False or vol) and (not False or adx) and gate(1,False,False,1,0,0,[0,0,0])
            self.assertEqual(result,baseline)

    def test_direction_confidence_warmup_and_confirmation_gates(self):
        means = [-.6,0,.6]
        self.assertTrue(gate(1,True,True,2,2,.55,means))
        self.assertTrue(gate(-1,True,True,0,2,.55,means))
        for ready,state,run,confidence in [(False,2,2,.9),(True,1,2,.9),(True,0,2,.9),(True,2,1,.9),(True,2,2,.54)]:
            self.assertFalse(gate(1,True,ready,state,run,confidence,means))
        self.assertFalse(gate(1,True,True,2,2,.99,[-.8,-.6,-.2]))
        self.assertFalse(gate(-1,True,True,0,2,.99,[.1,.3,.5]))

    def test_causal_refit_excludes_current_sample_and_bounds_history(self):
        model = Stream()
        xs = [math.sin(i/15) for i in range(240)]
        for i,x in enumerate(xs):
            result = model.execution(i,x)
            self.assertLessEqual(len(model.history),100)
            if i < 100: self.assertIsNone(result)
            if i in [100,125,150,175,200,225]: self.assertEqual(model.run,1)
        self.assertEqual([bar for bar,_ in model.fit_windows],[100,125,150,175,200,225])
        for bar,window in model.fit_windows: self.assertEqual(window,tuple(xs[bar-100:bar]))

    def test_future_observations_cannot_change_prior_outputs(self):
        xs = [math.sin(i/13)*.6 for i in range(300)]
        a,b = Stream(),Stream()
        before = [a.execution(i,x) for i,x in enumerate(xs[:220])]
        full = [b.execution(i,x) for i,x in enumerate(xs)]
        self.assertEqual(before,full[:220])

    def test_fill_callbacks_duplicate_bars_unconfirmed_and_disabled_do_not_train(self):
        a,b = Stream(),Stream()
        for i in range(160):
            x=math.sin(i/13)
            a.execution(i,x)
            b.execution(i,x,confirmed=False)
            b.execution(i,x,enabled=False)
            b.execution(i,x)
            b.execution(i,x,position=1)
            b.execution(i,x,position=0,closed=i+1)
            b.execution(i,x,closed=i+1)
        self.assertEqual(a.observations,b.observations)
        self.assertEqual(a.fit_windows,b.fit_windows)
        self.assertEqual(a.posterior,b.posterior)


class PineSourceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=(ROOT/'KAIROS_LC_Swing_Engine_V4_1_JXS_918.pine').read_text()
        cls.baseline=json.loads((ROOT/'tests/fixtures/kairos_v41_baseline.json').read_text())

    def test_exit_execution_and_old_defaults_preserved(self):
        start,end='// LOCK RISK LEVELS AFTER ACTUAL FILL','// STRATEGY STATISTICS TABLE'
        exit_code=self.s[self.s.index(start):self.s.index(end)]
        self.assertEqual(hashlib.sha256(exit_code.encode()).hexdigest(),self.baseline['exit_execution_sha256'])
        pattern=r'\b(?:bool|int|float|string)\s+(\w+)\s*=\s*input\.\w+\(\s*([^,\n]+)'
        before,after=self.baseline['input_defaults'],dict(re.findall(pattern,self.s))
        for key,value in before.items(): self.assertEqual(after[key],value,key)
        for side in ['long','short']:
            original=f'bool {side}V4FiltersPass = {side}LCFilterPass and {side}MLPredictionPass and {side}EMAEntryPass and {side}SMAEntryPass'
            self.assertIn(original+f' and lcEnvironmentPass and {side}HMMFilterPass',self.s)

    def test_pine_guard_and_current_sample_exclusion(self):
        self.assertIn('if useHMMFilter and barstate.isconfirmed and not lcFillRecalculation',self.s)
        start=self.s.index('if useHMMFilter and barstate.isconfirmed')
        block=self.s[start:self.s.index('bool hmmModelReady',start)]
        self.assertLess(block.index('f_hmmFit('),block.index('f_hmmForward('))
        self.assertLess(block.index('f_hmmForward('),block.index('array.push(hmmHistory'))
        self.assertIn('array.shift(hmmHistory)',block)
        self.assertIn('hmmLastProcessedBar != bar_index',block)
        self.assertIn('bool longHMMFilterPass = not useHMMFilter or',self.s)
        self.assertIn('bool shortHMMFilterPass = not useHMMFilter or',self.s)

    def test_original_lc_functions_and_independent_switches(self):
        for call in ['ml.regime_filter(ohlc4, lcRegimeThreshold, useLCRegimeFilter)',
                     'ml.filter_volatility(1, 10, useLCVolatilityFilter)',
                     'ml.filter_adx(close, 14, lcADXThreshold, useLCADXFilter)']:
            self.assertIn(call,self.s)
        for name in ['useLCRegimeFilter','useLCVolatilityFilter','useLCADXFilter','useHMMFilter']:
            self.assertRegex(self.s,rf'bool {name} = input.bool\(')
        module=self.s[self.s.index('string GROUP_REGIME'):self.s.index('// END V4.1 REGIME MODULE')]
        self.assertNotIn('useLCFilter',module)

    def test_table_bounds_and_balanced_delimiters(self):
        self.assertIn('     25,\n     bgcolor=statsBackground,',self.s)
        self.assertIn('table.clear(statsTable, 0, 0, 1, 24)',self.s)
        code=re.sub(r'"(?:[^"\\]|\\.)*"|//[^\n]*','',self.s)
        stack=[]
        for c in code:
            if c in '([': stack.append(c)
            if c in ')]': self.assertEqual(stack.pop(),{')':'(',']':'['}[c])
        self.assertEqual(stack,[])
        self.assertNotIn('text.align.',self.s)
        self.assertNotRegex(self.s,r'(?m)^array<int> order\b')


if __name__=='__main__': unittest.main()
