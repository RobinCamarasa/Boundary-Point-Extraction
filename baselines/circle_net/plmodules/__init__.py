from pathlib import Path
import sys


path = Path(__file__).parents[1] / 'CircleNet' / 'src' / 'lib'
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from .circle_net_modules import CarotidArteryChallengeCircleNet
