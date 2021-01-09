# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import pathlib
import math
import _pickle as cPickle

import re
import numpy as np
from tqdm import tqdm
import datetime
import shutil
import glob
import subprocess
import scipy.io.wavfile

from mmd.utils.MLogger import MLogger
from mmd.mmd.VmdData import VmdMorphFrame, VmdMotion
from mmd.mmd.PmxData import PmxModel
from mmd.mmd.VmdWriter import VmdWriter
from mmd.utils.MServiceUtils import get_file_encoding

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info('リップ生成処理開始: {0}', args.audio_file, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.audio_dir):
            logger.error("指定された音声ディレクトリパスが存在しません。\n{0}", args.audio_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(args.lyrics_file):
            logger.error("指定された歌詞ファイルパスが存在しません。\n{0}", args.lyrics_file, decoration=MLogger.DECORATION_BOX)
            return False

        logger.info("リップファイル生成開始", decoration=MLogger.DECORATION_LINE)

        # 歌詞ファイルをコピー
        lyrics_txts = []
        with open(args.lyrics_file, "r", encoding=get_file_encoding(args.lyrics_file)) as f:
            # 空白だけは許容
            lyrics_txts = [re.sub(r'( |　|\n|\r|！|\!|？|\?)', "", v) for v in f.readlines()]

        # そのまま結合
        lyric = "".join(lyrics_txts)

        # 全角カタカナはひらがなに変換
        _, katakana2hiragana = _make_kana_convertor()
        lyric = katakana2hiragana(lyric)

        # ひらがな以外はNG
        not_hira_list = re.findall(r'[^ぁ-んー]', lyric)
        if len(not_hira_list) > 0:
            # ひらがな以外はエラー
            logger.error("指定された歌詞ファイルにひらがな以外が含まれています。\n{0}\nエラー文字：{1}", args.lyrics_file, ",".join(not_hira_list), decoration=MLogger.DECORATION_BOX)
            return False

        with open(os.path.join(args.audio_dir, "vocals.txt"), "w", encoding='utf-8') as f:
            f.write(lyric)

        logger.info("音素分解開始（※しばらく無反応になります）", decoration=MLogger.DECORATION_LINE)

        # Perl スクリプトで音素分解
        popen = subprocess.Popen(["perl", "segment_julius.pl" , args.audio_dir], stdout=subprocess.PIPE)
        # 終了まで待つ
        popen.wait()

        logger.info("リップモーフ生成開始", decoration=MLogger.DECORATION_LINE)

        # wavを読み込み
        rate, data = scipy.io.wavfile.read(os.path.join(args.audio_dir, "vocals.wav"))
        # #16bitの音声ファイルのデータを-1から1に正規化
        normal_data = data / 32768
        #横軸（時間）の配列を作成　　#np.arange(初項, 等差数列の終点, 等差)
        time = np.arange(0, data.shape[0]/rate, 1/rate)

        lab_file = os.path.join(args.audio_dir, "vocals.lab")

        if os.path.getsize(lab_file) == 0:
            logger.error("音節取得に失敗しました。\n{0}", lab_file, ", ".join(not_hira_list), decoration=MLogger.DECORATION_BOX)
            return False

        lab_txts = []
        with open(lab_file, "r") as f:
            # 音素解析結果をそのまま読み込む
            lab_txts = [v.split() for v in f.readlines()]

        motion = VmdMotion()
        
        prev_fno = 0    # 前のキーフレ（母音でも子音でもOK）
        prev2_fno = 0   # 前の前のキーフレ（母音でも子音でもOK）
        prev_morph_name = None
        fno = 0
        start_ms = 0
        end_ms = 0
        for lidx, (start_ms_txt, end_ms_txt, syllable) in enumerate(tqdm(lab_txts, desc="Lip ...")):
            start_ms = float(start_ms_txt)
            end_ms = float(end_ms_txt)

            if start_ms * 1000 / 30 > fno + 1:
                # 開始秒数が区切りを超えている場合、インクリメント
                fno = int(round(start_ms * 30))

            # 音量データ生成
            now_values = []    # 現在の範囲の音量
            
            for t, d in zip(time, normal_data):
                if start_ms <= t <= end_ms:
                    now_values.append(d)

            # 母音の最大値を口の大きさとする
            nratio = max(0, min(1, np.max(now_values)))

            for vowel, morph_name in [("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お")]:
                if syllable.startswith(vowel):
                    # 前の前のと前のキーフレの間をクリア（今回の開始）
                    p2mf = VmdMorphFrame(max(0, int(prev2_fno + round((prev_fno - prev2_fno) / 2))))
                    p2mf.set_name(morph_name)
                    p2mf.ratio = 0
                    motion.regist_mf(p2mf, p2mf.name, p2mf.fno)

                    # 前のと今回のキーフレの間をクリア（前回の終了）
                    if prev_morph_name and lidx > 0:
                        pmf = VmdMorphFrame(int(prev_fno + round((fno - prev_fno) / 2)))
                        pmf.set_name(prev_morph_name)
                        pmf.ratio = 0
                        motion.regist_mf(pmf, pmf.name, pmf.fno)

                    # 母音は最大値
                    vmf = VmdMorphFrame(fno)
                    vmf.set_name(morph_name)
                    vmf.ratio = nratio
                    motion.regist_mf(vmf, vmf.name, vmf.fno)

                    # 前の母音として保持
                    prev_morph_name = morph_name

                    break
            
            prev2_fno = prev_fno
            prev_fno = fno
        
        # 最後を閉じる
        fno = int(math.ceil(end_ms * 30))
        for vowel, morph_name in [("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お")]:
            vmf = VmdMorphFrame(fno)
            vmf.set_name(morph_name)
            vmf.ratio = 0
            motion.regist_mf(vmf, vmf.name, vmf.fno)

        logger.info('リップ生成処理終了: {0}', args.audio_dir, decoration=MLogger.DECORATION_BOX)

        logger.info("モーション生成開始", decoration=MLogger.DECORATION_LINE)

        process_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        motion_path = os.path.join(args.audio_dir, "{0}_lip_{1}.vmd".format(os.path.basename(lab_file), process_datetime))

        model = PmxModel()
        model.name = "リップモデル"

        writer = VmdWriter(model, motion, motion_path)
        writer.write()

        logger.info("モーション生成終了: {0}", motion_path, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("リップ生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
 

def _make_kana_convertor():
    """ひらがな⇔カタカナ変換器を作る"""
    kata = {
        'ア':'あ', 'イ':'い', 'ウ':'う', 'エ':'え', 'オ':'お',
        'カ':'か', 'キ':'き', 'ク':'く', 'ケ':'け', 'コ':'こ',
        'サ':'さ', 'シ':'し', 'ス':'す', 'セ':'せ', 'ソ':'そ',
        'タ':'た', 'チ':'ち', 'ツ':'つ', 'テ':'て', 'ト':'と',
        'ナ':'な', 'ニ':'に', 'ヌ':'ぬ', 'ネ':'ね', 'ノ':'の',
        'ハ':'は', 'ヒ':'ひ', 'フ':'ふ', 'ヘ':'へ', 'ホ':'ほ',
        'マ':'ま', 'ミ':'み', 'ム':'む', 'メ':'め', 'モ':'も',
        'ヤ':'や', 'ユ':'ゆ', 'ヨ':'よ', 'ラ':'ら', 'リ':'り',
        'ル':'る', 'レ':'れ', 'ロ':'ろ', 'ワ':'わ', 'ヲ':'を',
        'ン':'ん',
        
        'ガ':'が', 'ギ':'ぎ', 'グ':'ぐ', 'ゲ':'げ', 'ゴ':'ご',
        'ザ':'ざ', 'ジ':'じ', 'ズ':'ず', 'ゼ':'ぜ', 'ゾ':'ぞ',
        'ダ':'だ', 'ヂ':'ぢ', 'ヅ':'づ', 'デ':'で', 'ド':'ど',
        'バ':'ば', 'ビ':'び', 'ブ':'ぶ', 'ベ':'べ', 'ボ':'ぼ',
        'パ':'ぱ', 'ピ':'ぴ', 'プ':'ぷ', 'ペ':'ぺ', 'ポ':'ぽ',
        
        'ァ':'ぁ', 'ィ':'ぃ', 'ゥ':'ぅ', 'ェ':'ぇ', 'ォ':'ぉ',
        'ャ':'ゃ', 'ュ':'ゅ', 'ョ':'ょ',
        'ヴ':'う', 'ッ':'っ', 'ヰ':'い', 'ヱ':'え',
        }
    
    # ひらがな → カタカナ のディクショナリをつくる
    hira = dict([(v, k) for k, v in kata.items() ])
    
    re_hira2kata = re.compile("|".join(map(re.escape, hira)))
    re_kata2hira = re.compile("|".join(map(re.escape, kata)))
    
    def _hiragana2katakana(text):
        return re_hira2kata.sub(lambda x: hira[x.group(0)], text)
    
    def _katakana2hiragana(text):
        return re_kata2hira.sub(lambda x: kata[x.group(0)], text)
    
    return (_hiragana2katakana, _katakana2hiragana)
