from typing import Generator
from io import BufferedRandom


def generator_tmp_file(
    tmpfile: BufferedRandom, i: int = 0, encoding: str = "ascii"
) -> Generator[float, None, None]:
    while True:
        line: bytes = tmpfile.readline()
        if len(line) == 0 or line is EOFError:
            break

        if not line:
            yield 0.0  # offset past EOF
            continue
        # take bytes up to first space (or whole line), strip newline
        i = line.find(b" ")
        # take byets from space onwards so only the probability
        token = line[i + 1 :] if i != -1 else line.rstrip(b"\r\n")
        # Conversion needed to maintain stability this is according to the ccs15 paper on the montecarlo estimation
        """
            'The probabilities that we compute can be very small and may underflow:
            to avoid such problems, we store and compute the base-2 logarithms of probabilities rather than probabilities themselves'

            ccs15 page 5, implementation details https://www.dcs.gla.ac.uk/~maurizio/Publications/ccs15.pdf
        """

        yield float(token.decode(encoding, errors="replace"))


def fast_estimate(): ...
