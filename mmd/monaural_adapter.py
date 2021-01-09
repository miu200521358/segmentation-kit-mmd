import ffmpeg

import os
from spleeter.audio.ffmpeg import FFMPEGProcessAudioAdapter, _check_ffmpeg_install, _to_ffmpeg_codec
from spleeter import SpleeterError
from spleeter.utils.logging import get_logger

class FFMPEGMonauralProcessAudioAdapter(FFMPEGProcessAudioAdapter):

    def save(
            self, path, data, sample_rate,
            codec=None, bitrate=None):
        """ Write waveform data to the file denoted by the given path
        using FFMPEG process.

        :param path: Path of the audio file to save data in.
        :param data: Waveform data to write.
        :param sample_rate: Sample rate to write file in.
        :param codec: (Optional) Writing codec to use.
        :param bitrate: (Optional) Bitrate of the written audio file.
        :raise IOError: If any error occurs while using FFMPEG to write data.
        """
        _check_ffmpeg_install()
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            raise SpleeterError(f'output directory does not exists: {directory}')
        get_logger().debug('Writing file %s', path)
        input_kwargs = {'ar': sample_rate, 'ac': data.shape[1]}
        output_kwargs = {'ar': 16000, 'ac': 1}
        if bitrate:
            output_kwargs['audio_bitrate'] = bitrate
        if codec is not None and codec != 'wav':
            output_kwargs['codec'] = _to_ffmpeg_codec(codec)
        process = (
            ffmpeg
            .input('pipe:', format='f32le', **input_kwargs)
            .output(path, **output_kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True, pipe_stderr=True, quiet=True))
        try:
            process.stdin.write(data.astype('<f4').tobytes())
            process.stdin.close()
            process.wait()
        except IOError:
            raise SpleeterError(f'FFMPEG error: {process.stderr.read()}')
        get_logger().info('File %s written succesfully', path)
    
