"""Launcher-contract checks; no model dependencies and no actual experiments."""
import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'scripts/run_energygradmem_on_squad.sh'
SUPPORT = ROOT / 'scripts/run_support_distillation_on_squad.sh'


class SupportLauncherTests(unittest.TestCase):
    def test_preserves_source_variables_arguments_and_lifecycle(self):
        base, support = BASE.read_text(), SUPPORT.read_text()
        variables = lambda text: set(re.findall(r'^([A-Z][A-Z0-9_]*)=', text, re.M))
        flags = lambda text: set(re.findall(r'--([A-Za-z][A-Za-z0-9_]*)', text))
        self.assertFalse(variables(base) - variables(support))
        self.assertFalse(flags(base) - flags(support))
        for operation in ('collect_env_state.sh', 'prepare_locked_run', 'find_free_port',
                          'print_run_header', 'start_run_timer', 'PIPESTATUS[0]',
                          'finalize_locked_run', 'for N in "${N_VALUES[@]}"'):
            self.assertIn(operation, support)
        # Preserve the exact inherited assignments, except the algorithm-specific
        # default objective. This detects accidental simplifications of settings.
        for line in base.splitlines():
            if re.match(r'^[A-Z][A-Z0-9_]*=', line) and not line.startswith(('WRITE_OBJECTIVE=', 'RUN_NAME=')):
                self.assertIn(line, support)

    def test_every_python_argument_has_explicit_launcher_forwarding(self):
        tree = ast.parse((ROOT / 'run_support_distillation_on_squad.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'ExperimentArgs')
        fields = {node.target.id for node in cls.body if isinstance(node, ast.AnnAssign)}
        flags = set(re.findall(r'--([A-Za-z][A-Za-z0-9_]*)', SUPPORT.read_text()))
        self.assertFalse(fields - flags, f'Missing CLI forwarding: {fields - flags}')

    def test_support_variables_reach_command_and_run_name(self):
        # Replace only external lifecycle utilities and Accelerate in an isolated
        # temporary directory. Never run the real collector, model or GPU launcher.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            script = path / SUPPORT.name
            script.write_text(SUPPORT.read_text())
            (path / 'collect_env_state.sh').write_text('''
prepare_locked_run() { RUN_LOCK_LOG="$CAPTURE.log"; RUN_LOCK_DIR="$CAPTURE.lock"; }
find_free_port() { echo 12345; }
print_run_header() { :; }
start_run_timer() { :; }
finalize_locked_run() { return "$4"; }
''')
            fake = path / 'accelerate'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=f'{path}:{os.environ["PATH"]}',
                       CAPTURE=str(path / 'arguments'), N_SUPPORT_QUESTIONS='3',
                       INNER_DISTILL_QUERY_WEIGHT='0.2', INNER_DISTILL_TARGET_WEIGHT='0.8',
                       INNER_RECONSTRUCTION_WEIGHT='0.5', READ_CE_WEIGHT='0.7',
                       DISTILL_LOGITS_TARGET_WEIGHT='0.4', WARMUP_STEPS='321')
            subprocess.run(['bash', str(script)], cwd=path, env=env, check=True,
                           capture_output=True, text=True)
            args = (path / 'arguments').read_text().splitlines()
            for flag, value in [('n_support_questions', '3'), ('inner_distill_query_weight', '0.2'),
                                ('inner_distill_target_weight', '0.8'), ('inner_reconstruction_weight', '0.5'),
                                ('read_ce_weight', '0.7'), ('distill_logits_target_weight', '0.4'),
                                ('warmup_steps', '321')]:
                self.assertEqual(args[args.index('--' + flag) + 1], value)
            run_path = args[args.index('--exp_path') + 1]
            for component in ('support3', 'iklq0.2', 'iklt0.8', 'ice0.5', 'ce0.7', 'klt0.4', 'warm321'):
                self.assertIn(component, run_path)
            self.assertIn('run_support_distillation_on_squad.py', args)
            self.assertNotIn('run_gradmemgpt_on_squad.py', args)


if __name__ == '__main__':
    unittest.main()
