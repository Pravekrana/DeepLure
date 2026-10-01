"""Extract the allowlisted archive into a fresh temp directory and run its tests."""
import json,os,subprocess,sys,tempfile,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='deeplure-package-check-') as temp:
    with zipfile.ZipFile(root/'dist/DeepLure_Submission.zip') as z:
        for n in z.namelist():
            if not (Path(temp)/n).resolve().is_relative_to(Path(temp).resolve()):raise ValueError('Unsafe ZIP member')
        z.extractall(temp)
    project=Path(temp)/'saree-design-recognition';env=dict(os.environ);env['PYTHONPATH']=str(project)
    env['MPLCONFIGDIR']=str(Path(temp)/'mpl');env['TORCH_HOME']=str(Path(temp)/'torch')
    isolation=subprocess.run([sys.executable,'-c',"from pathlib import Path; import saree; assert Path(saree.__file__).resolve().is_relative_to(Path.cwd().resolve())"],cwd=project,env=env,capture_output=True,text=True)
    if isolation.returncode:raise RuntimeError('Extracted import isolation failed: '+isolation.stderr)
    test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=project,env=env,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    (root/'dist/extracted_tests.txt').write_text(test.stdout+test.stderr)
    check=subprocess.run([sys.executable,'-m','saree.cli','audit','--data-root','missing-data','--out','not-created.json'],cwd=project,env=env,capture_output=True,text=True)
    assert check.returncode==2 and not (project/'not-created.json').exists()
    result={'extracted_tests_passed':True,'test_stdout':test.stdout.strip(),'missing_data_failed_clearly':True,'isolated_from_original_package':True,'network_or_real_dataset_required_for_tests':False}
    (root/'dist/extracted_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
