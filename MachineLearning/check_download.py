"""Run from Strict: python MachineLearning/check_download.py [native]."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import sys

root = Path(__file__).resolve().parent.parent
class Missing(BaseHTTPRequestHandler):
	def do_GET(self):
		self.send_error(404)
	def log_message(self, *arguments):
		pass
with tempfile.TemporaryDirectory(prefix="DownloadCheck", dir=root / "Examples") as folder:
	server = ThreadingHTTPServer(("127.0.0.1", 0), Missing)
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		path = Path(folder) / "MarkovText.strict"
		text = (root / "MachineLearning/MarkovText.strict").read_text(encoding="utf-8")
		text = text.replace('has logger', 'has logger\nhas chain MachineLearning/Markov', 1)
		text = text.replace('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
			f'http://127.0.0.1:{server.server_port}/missing')
		target = Path(folder) / "missing.txt"
		text = text.replace('MachineLearning/shakespeare.txt', target.as_posix())
		path.write_text(text, encoding="utf-8")
		command = [str(root / "strict.exe"), str(path)]
		if "native" in sys.argv:
			built = subprocess.run([str(root / "strict.exe"), "build", str(path)], cwd=root,
				capture_output=True, text=True, timeout=60)
			assert built.returncode == 0, built.stdout + built.stderr
			command = [str(path.with_suffix(".exe"))]
		result = subprocess.run(command, cwd=root,
			capture_output=True, text=True, timeout=60)
		assert result.returncode != 0 and "download failed" in result.stdout.lower(), result.stdout + result.stderr
		assert not target.exists(), "Failed download was published as complete"
		print("MarkovText reports the download failure without publishing a partial corpus")
	finally:
		server.shutdown()
		server.server_close()
		thread.join(timeout=5)
