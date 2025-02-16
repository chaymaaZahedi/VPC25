#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################







import numpy as np
from pydub import AudioSegment
import soundfile as sf  # Pour convertir l'audio en tableau NumPy

def anonymize(input_audio_path):
    """
    Anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32.
    sr : int
        The sample rate of the processed audio.
    """
    # Paramètres par défaut pour le pitch shifting et le time-stretching
    pitch_shift_steps = 2
    speed_change = 1.1

    # Charger l'audio avec pydub
    audio = AudioSegment.from_file(input_audio_path)

    # Appliquer le pitch shifting
    new_sample_rate = int(audio.frame_rate * (5.0 ** (pitch_shift_steps / 8.0)))
    shifted_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    shifted_audio = shifted_audio.set_frame_rate(audio.frame_rate)

    # Appliquer le time-stretching (changer la vitesse sans affecter la hauteur)
    shifted_audio = shifted_audio.speedup(playback_speed=speed_change)

    # Convertir l'audio en tableau NumPy et en np.float32
    samples = np.array(shifted_audio.get_array_of_samples())
    audio_normalized = samples.astype(np.float32) / np.iinfo(samples.dtype).max  # Normalisation
    sr = shifted_audio.frame_rate

    return audio_normalized, sr