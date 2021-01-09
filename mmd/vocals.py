# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import pathlib
import _pickle as cPickle

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import datetime
import shutil

from spleeter.separator import Separator
from spleeter.audio.adapter import get_audio_adapter
from mmd.utils.MLogger import MLogger
from mmd.monaural_adapter import FFMPEGMonauralProcessAudioAdapter

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info('音声認識処理開始: {0}', args.audio_file, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.audio_file):
            logger.error("指定された音声ファイルパスが存在しません。\n{0}", args.audio_file, decoration=MLogger.DECORATION_BOX)
            return False, None

        # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
        base_path = str(pathlib.Path(args.audio_file).parent) if not args.parent_dir else args.parent_dir

        if len(args.parent_dir) > 0:
            process_audio_dir = base_path
        else:
            process_audio_dir = os.path.join(base_path, "{0}_{1:%Y%m%d_%H%M%S}".format(os.path.basename(args.audio_file).replace('.', '_'), datetime.datetime.now()))

        # 既存は削除
        if os.path.exists(process_audio_dir):
            shutil.rmtree(process_audio_dir)

        # フォルダ生成
        os.makedirs(process_audio_dir)

        audio_adapter = FFMPEGMonauralProcessAudioAdapter()
        sample_rate = 44100
        waveform, _ = audio_adapter.load(args.audio_file, sample_rate=sample_rate)

        # 音声と曲に分離
        separator = Separator('spleeter:2stems')

        # Perform the separation :
        prediction = separator.separate(waveform)

        # 音声データ
        vocals = prediction['vocals']

        vocals_wav_path = f"{process_audio_dir}/vocals.wav"

        # 一旦wavとして保存
        audio_adapter.save(vocals_wav_path, vocals, sample_rate, "wav")

        logger.info('音声認識処理終了: {0}', process_audio_dir, decoration=MLogger.DECORATION_BOX)

        return True, process_audio_dir
    except Exception as e:
        logger.critical("音声認識で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None
 

 