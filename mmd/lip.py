# -*- coding: utf-8 -*-
import os
import math

import re
import numpy as np
from tqdm import tqdm
import datetime
import subprocess

from spleeter.audio.adapter import get_default_audio_adapter
from mmd.utils.MLogger import MLogger
from mmd.mmd.VmdData import VmdMorphFrame, VmdMotion
from mmd.mmd.PmxData import PmxModel
from mmd.mmd.VmdWriter import VmdWriter
from mmd.utils.MServiceUtils import get_file_encoding
from mmd.monaural_adapter import FFMPEGMonauralProcessAudioAdapter

logger = MLogger(__name__, level=1)

SEPARATOR = "----------"

def execute(args):
    try:
        logger.info('リップ生成処理開始: {0}', args.audio_dir, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.audio_dir):
            logger.error("指定された音声ディレクトリパスが存在しません。\n{0}", args.audio_dir, decoration=MLogger.DECORATION_BOX)
            return False

        if not os.path.exists(args.lyrics_file):
            logger.error("指定された歌詞ファイルパスが存在しません。\n{0}", args.lyrics_file, decoration=MLogger.DECORATION_BOX)
            return False
        
        vocal_audio_file = os.path.join(args.audio_dir, 'vocals.wav')

        logger.info("リップファイル生成開始", decoration=MLogger.DECORATION_LINE)

        # exoヘッダテンプレートを読み込み
        exo_header_txt = None
        with open(os.path.join("config", "exo.head.txt"), "r", encoding='shift-jis') as f:
            exo_header_txt = f.read()

        # exo文字テンプレートを読み込み
        exo_chara_txt = None
        with open(os.path.join("config", "exo.chara.txt"), "r", encoding='shift-jis') as f:
            exo_chara_txt = f.read()

        # 歌詞ファイルをコピー
        separates = []
        full_lyrics_txts = []
        with open(args.lyrics_file, "r", encoding=get_file_encoding(args.lyrics_file)) as f:
            # 空白だけは許容
            # 連続した改行は分割文字列に置換
            lyric = ""
            for v in f.readlines():
                if re.fullmatch(r'^(\d?\d)\:(\d\d).(\d\d\d)-(\d?\d)\:(\d\d).(\d\d\d)$', v.strip()):
                    m = re.match(r'^(\d?\d)\:(\d\d).(\d\d\d)-(\d?\d)\:(\d\d).(\d\d\d)$', v.strip())

                    # 開始秒数
                    separate_start_sec = float(m.group(1)) * 60 + float(m.group(2)) + float(m.group(3)) * 0.001
                    # 終了秒数
                    separate_end_sec = float(m.group(4)) * 60 + float(m.group(5)) + float(m.group(6)) * 0.001
                    # 追加
                    separates.append((separate_start_sec, separate_end_sec, v.strip()))
                else:
                    if len(v.strip()) == 0:
                        # 改行のみの場合、追加
                        full_lyrics_txts.append(lyric)
                        lyric = ""
                    elif re.match(r'\d', v.strip()) is not None and lyric == "":
                        logger.warning("秒数区切りの書式が間違っています。\n入力文字列: {0}\n{1}", v.strip(), args.lyrics_file, decoration=MLogger.DECORATION_BOX)
                    else:
                        # 普通の文字列は結合だけしておく
                        lyric += re.sub(r'\n', '', re.sub(r'(！|\!|？|\?| |　|、|。)+', "", v))
            
            full_lyrics_txts.append(lyric)

        if len(separates) != len(full_lyrics_txts):
            logger.error("歌詞と秒数区切りのペアが正しく設定されていません。\nひとつ上に具体的なエラー箇所が記されてますので、確認してください。\n{0}", args.lyrics_file, decoration=MLogger.DECORATION_BOX)
            return False
        
        prev_separate_start_sec = 0
        for sidx, (separate_start_sec, separate_end_sec, separate_txt) in enumerate(separates):
            if separate_start_sec > separate_end_sec:
                logger.error("{0}番目のブロックが、終了秒に開始秒より前の値が設定されています。\n終了秒数: {1}, 開始秒数: {2}({3}), \n{4}", \
                             sidx, separate_end_sec, separate_start_sec, separate_txt, args.lyrics_file, decoration=MLogger.DECORATION_BOX)
                return False

            if sidx > 0 and separate_start_sec < prev_separate_start_sec:
                logger.error("{0}番目のブロックが、ひとつ前のブロックの開始秒より前の値が設定されています。\n前回の開始秒数: {1}, 今回の開始秒数: {2}({3}), \n{4}", \
                             sidx, prev_separate_start_sec, separate_start_sec, separate_txt, args.lyrics_file, decoration=MLogger.DECORATION_BOX)
                return False
            prev_separate_start_sec = separate_start_sec

        # 全角カタカナはひらがなに変換
        full_lyric = "".join(full_lyrics_txts)
        full_lyric = katakana2hiragana(full_lyric)

        # ひらがな以外はNG
        not_hira_list = re.findall(r'[^っぁ-んー\-{10}( sp )]', full_lyric)
        if len(not_hira_list) > 0:
            # ひらがな以外はエラー
            logger.error("指定された歌詞ファイルに全角カナ・ひらがな以外が含まれています。\n{0}\nエラー文字：{1}", args.lyrics_file, ",".join(not_hira_list), decoration=MLogger.DECORATION_BOX)
            return False

        # wavを読み込み
        audio_adapter = FFMPEGMonauralProcessAudioAdapter()
        data, org_rate = audio_adapter.load(vocal_audio_file, sample_rate=16000)
        org_rate = int(org_rate)
        #横軸（時間）の配列を作成
        time = np.arange(0, data.shape[0]/org_rate, 1/org_rate)

        end_fno = int(math.ceil(time[-1] * 30))

        # モーションデータ
        motion = VmdMotion()

        # exoデータ
        process_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exo_file_path = os.path.join(args.audio_dir, f"{process_datetime}_lyric.exo")
        lyric_exo_f = open(exo_file_path, "w", encoding='shift-jis')
        lyric_exo_f.write(exo_header_txt.replace("<<length>>", str(end_fno)))

        start_fno = 0
        fno = 0
        now_exo_chara_txt = ""
        fidx = 0
        end_s = 0

        # WAV形式で 16kHz, 16bit, PCM（無圧縮）形式
        rate = 16000

        is_failure = False

        for tidx, ((separate_start_sec, separate_end_sec, separate_txt), lyrics) in enumerate(zip(separates, full_lyrics_txts)):
            tidx_dir_name = f"{tidx:03}"

            hira_lyric = katakana2hiragana(lyrics)

            logger.info("【No.{0}】入力歌詞:\n{1}", tidx, lyrics)

            # ディレクトリ作成
            os.makedirs(os.path.join(args.audio_dir, tidx_dir_name), exist_ok=True)

            start_fno = int(separate_start_sec * 30)
            fno = start_fno

            if len(hira_lyric) > 300:
                # 300文字以上はスルー（spを想定して少し幅を持たせてある）
                logger.warning("【No.{0}】250文字以上の区間は一度に出力できないため、処理をスキップします。分割してください。\n{1}", f'{tidx:03}', hira_lyric, decoration=MLogger.DECORATION_BOX)
                is_failure = True
                continue
            
            block_audio_file = os.path.join(args.audio_dir, tidx_dir_name, 'block.wav')

            # wavファイルの一部を分割保存
            audio_adapter.save(block_audio_file, data[round(separate_start_sec*org_rate):(round(separate_end_sec*org_rate)-1)], rate, "wav")

            # 分割データを再読み込み
            sep_data, rate = audio_adapter.load(block_audio_file, sample_rate=16000)
            rate = int(rate)
            #横軸（時間）の配列を作成
            time = np.arange(0, sep_data.shape[0]/rate, 1/rate)

            # 分割した歌詞を出力
            with open(os.path.join(args.audio_dir, tidx_dir_name, 'block.txt'), "w", encoding='utf-8') as f:
                f.write(hira_lyric)

            logger.info("【No.{0}】音素分解開始", f'{tidx:03}', decoration=MLogger.DECORATION_LINE)

            # Perl スクリプトで音素分解
            popen = subprocess.Popen(["perl", "segment_julius.pl" , os.path.join(args.audio_dir, tidx_dir_name)], stdout=subprocess.PIPE)
            try:
                # 終了まで待つ(30秒でタイムアウト)
                popen.wait(timeout=30)
            except subprocess.TimeoutExpired as e:
                try:
                    popen.kill()
                except Exception:
                    pass
                is_failure = True
                logger.warning("【No.{0}】音素分解に失敗しました。", f'{tidx:03}', ", ".join(not_hira_list), decoration=MLogger.DECORATION_BOX)
                continue

            logger.info("【No.{0}】リップモーフ生成開始", f'{tidx:03}', decoration=MLogger.DECORATION_LINE)

            lab_file = os.path.join(args.audio_dir, tidx_dir_name, f'block.lab')

            if not os.path.exists(lab_file) or os.path.getsize(lab_file) == 0:
                logger.warning("【No.{0}】音節取得に失敗しました。\n{1}", f'{tidx:03}', lab_file, ", ".join(not_hira_list), decoration=MLogger.DECORATION_BOX)
                is_failure = True
                continue

            lab_txts = []
            with open(lab_file, "r") as f:
                # 音素解析結果をそのまま読み込む
                lab_txts = [v.split() for v in f.readlines()]
            
            prev_start_s = 0
            prev_syllable = ""
            prev_morph_name = ""
            now_morph_name = ""
            VOWELS = ["a", "i", "u", "e", "o", "a:", "i:", "u:", "e:", "o:"]
            ENDS = ["N"]

            for lidx, (start_s_txt, end_s_txt, syllable) in enumerate(lab_txts):
                start_s =  min(len(lab_txts), max(0, float(start_s_txt) - 0.05))
                end_s = min(len(lab_txts), max(0, float(end_s_txt) + 0.05))

                # キーフレは開始と終了の間
                now_start_fno = start_fno + round(prev_start_s * 30)
                now_end_fno = start_fno + round(end_s * 30)

                now_ratios = []
                for vowel, morph_name in [("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お")]:
                    if syllable.startswith(vowel):
                        now_morph_name = morph_name

                if syllable in ENDS:
                    # んの場合、閉じる
                    vsmf = VmdMorphFrame(max(0, now_start_fno))
                    vsmf.set_name(now_morph_name)
                    vsmf.ratio = 0
                    motion.regist_mf(vsmf, vsmf.name, vsmf.fno)

                    vemf = VmdMorphFrame(max(0, now_end_fno))
                    vemf.set_name(now_morph_name)
                    vemf.ratio = 0
                    motion.regist_mf(vemf, vemf.name, vemf.fno)
                    
                    now_morph_name = "ん"                    
                elif syllable in VOWELS:
                    now_start_s = start_s if prev_syllable in VOWELS or prev_syllable in ENDS else prev_start_s
                    now_vs = []

                    if args.threshold < 1:
                        # 前が母音もしくは終端の場合、現在から始める。子音の場合は前から繋げる
                        s =  now_start_s
                        while s < end_s:
                            # 音量範囲のINDEX
                            rs = round(s * rate)
                            rf = min(len(sep_data), round((s + 1 / 30) * rate) - 1)

                            vs = sep_data[rs:rf]
                            f = round(s * 30) + start_fno

                            if len(vs) > 0:
                                # 母音の変動
                                vsmf = VmdMorphFrame(max(0, f))
                                vsmf.set_name(now_morph_name)
                                # 端っこは小さめにする
                                vsmf.ratio = min(1, float(np.max(vs)) * min(((min(2, ((s - now_start_s) * 30)) / 2), ((min(2, (end_s - s) * 30)) / 2))))
                                motion.regist_mf(vsmf, vsmf.name, vsmf.fno)
                                now_ratios.append(str(round(vsmf.ratio, 3)))

                            s += 1 / 30

                        if prev_morph_name != now_morph_name:
                            # 母音の開始(上書き)
                            # 母音が同じ場合、既にratioが入っているので入れない
                            mf = motion.calc_mf(now_morph_name, now_start_fno)
                            mf.ratio = 0
                            motion.regist_mf(mf, mf.name, mf.fno)

                        # 母音の終了
                        # 前の母音が残っていたら終了
                        for m in ["あ", "い", "う", "え", "お"]:
                            mf = motion.calc_mf(m, now_end_fno)
                            if mf.ratio != 0:
                                mf.ratio = 0
                                motion.regist_mf(mf, mf.name, mf.fno)

                    elif args.threshold == 1:
                        # 1の場合、最高値を登録する

                        # 区間内に同じモーフで違うキーがある場合、削除                        
                        s =  now_start_s
                        while s < end_s:
                            f = round(s * 30) + start_fno
                            mf = motion.calc_mf(now_morph_name, f)
                            if mf.key:
                                del motion.morphs[now_morph_name][f]
                            s += 1 / 30

                        rs = round(now_start_s * rate)
                        rf = round(end_s * rate)

                        vs = sep_data[rs:rf]
                        fs = min(now_start_fno + 2, now_end_fno - 2)
                        fe = max(now_end_fno - 2, now_start_fno + 2)

                        if len(vs) > 0:
                            # 台形になるように、開始と終了に同じ値
                            vsmf = VmdMorphFrame(max(0, fs))
                            vsmf.set_name(now_morph_name)
                            vsmf.ratio = min(1, float(np.max(vs)))
                            motion.regist_mf(vsmf, vsmf.name, vsmf.fno)
                            now_ratios.append(f'{fs}:{round(vsmf.ratio, 3)}')

                            vemf = VmdMorphFrame(max(0, fe))
                            vemf.set_name(now_morph_name)
                            vemf.ratio = min(1, float(np.max(vs)))
                            motion.regist_mf(vemf, vemf.name, vemf.fno)
                            now_ratios.append(f'{fe}:{round(vemf.ratio, 3)}')
                        
                        vsmf = VmdMorphFrame(now_start_fno)
                        vsmf.set_name(now_morph_name)
                        vsmf.ratio = 0
                        motion.regist_mf(vsmf, vsmf.name, vsmf.fno)

                        vsmf = VmdMorphFrame(now_end_fno)
                        vsmf.set_name(now_morph_name)
                        vsmf.ratio = 0
                        motion.regist_mf(vsmf, vsmf.name, vsmf.fno)

                if syllable in VOWELS or syllable in ENDS:
                    # exoデータを出力
                    now_exo_chara_txt = str(exo_chara_txt)
                    now_chara = prev_syllable + syllable if prev_syllable not in VOWELS else syllable
                    # ひらがな変換
                    now_kana = romaji2hiragana(now_chara)
                    # 長音追加
                    now_kana = now_kana.replace(":", "ー")
                    # ユニコードエスケープ
                    now_uni_chara =to_unicode_escape(now_kana)
                    layer = int(fidx % 3) + 1
                    logger.test(f"fno: {fno}, index: {fidx}, start_fno: {now_start_fno}, layer: {layer}, text: {romaji2hiragana(now_chara)}, uni: {now_uni_chara}")
                    for format_txt, value in [("<<index>>", fidx), ("<<start_fno>>", now_start_fno), ("<<end_fno>>", now_end_fno), ("<<layer>>", layer), \
                                                ("<<encoded_txt>>", now_uni_chara.ljust(4096, '0'))]:
                        now_exo_chara_txt = now_exo_chara_txt.replace(format_txt, str(value))
                    lyric_exo_f.write(now_exo_chara_txt)
                    fidx += 1

                    logger.info(f"[{tidx}-{lidx}][{now_kana}:{now_morph_name}] start: {now_start_fno}({round(start_s,4)}), range: {','.join(now_ratios)} end: {now_end_fno}({round(end_s,4)})")

                prev_start_s = start_s
                prev_syllable = syllable
                prev_morph_name = now_morph_name

            logger.info("【No.{0}】リップモーフ生成終了", f'{tidx:03}', decoration=MLogger.DECORATION_LINE)

        if 0 < args.threshold < 1:
            logger.info("不要モーフキー削除処理", decoration=MLogger.DECORATION_LINE)
            for morph_name in tqdm(['あ', 'い', 'う', 'え', 'お']):
                motion.remove_unnecessary_mf(-1, morph_name, threshold=args.threshold)

        logger.info("モーション生成開始", decoration=MLogger.DECORATION_LINE)

        motion_path = os.path.join(args.audio_dir, f"{process_datetime}_lip.vmd")

        model = PmxModel()
        model.name = "リップモデル"

        writer = VmdWriter(model, motion, motion_path)
        writer.write()

        logger.info("モーション生成終了: {0}", motion_path, decoration=MLogger.DECORATION_BOX)

        lyric_exo_f.close()
        logger.info("exoファイル生成終了: {0}", exo_file_path, decoration=MLogger.DECORATION_BOX)

        if is_failure:
            logger.warning("モーフ生成に失敗してる区間があります。ログを確認してください。", decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("リップ生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def to_unicode_escape(txt):
    escape_txt = ""
    for c in txt:
        escape_chara = c.encode('unicode_escape').decode('utf-8')
        escape_txt += escape_chara[4:6]
        escape_txt += escape_chara[2:4]

    return escape_txt


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
        'ヴ':'う', 'ッ':'っ', 'ー': 'ー'
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

hiragana2katakana, katakana2hiragana = _make_kana_convertor()

def _make_romaji_convertor():
    """ローマ字⇔かな変換器を作る"""
    master = {
        'la':'ァ', 'li':'ィ', 'lu':'ゥ', 'le':'ェ', 'lo':'ォ',
        'a'  :'ア', 'i'  :'イ', 'u'  :'ウ', 'e'  :'エ', 'o'  :'オ',
        'ka' :'カ', 'ki' :'キ', 'ku' :'ク', 'ke' :'ケ', 'ko' :'コ',
        'sa' :'サ', 'shi':'シ', 'su' :'ス', 'se' :'セ', 'so' :'ソ',
        'ta' :'タ', 'chi':'チ', 'tsu' :'ツ', 'te' :'テ', 'to' :'ト',
        'na' :'ナ', 'ni' :'ニ', 'nu' :'ヌ', 'ne' :'ネ', 'no' :'ノ',
        'ha' :'ハ', 'hi' :'ヒ', 'fu' :'フ', 'he' :'ヘ', 'ho' :'ホ',
        'ma' :'マ', 'mi' :'ミ', 'mu' :'ム', 'me' :'メ', 'mo' :'モ',
        'ya' :'ヤ', 'yu' :'ユ', 'yo' :'ヨ',
        'ra' :'ラ', 'ri' :'リ', 'ru' :'ル', 're' :'レ', 'ro' :'ロ',
        'wa' :'ワ', 'wo' :'ヲ', 'n'  :'ン', 'vu' :'ヴ',
        'ga' :'ガ', 'gi' :'ギ', 'gu' :'グ', 'ge' :'ゲ', 'go' :'ゴ',
        'za' :'ザ', 'ji' :'ジ', 'zu' :'ズ', 'ze' :'ゼ', 'zo' :'ゾ',
        'da' :'ダ', 'di' :'ヂ', 'du' :'ヅ', 'de' :'デ', 'do' :'ド',
        'ba' :'バ', 'bi' :'ビ', 'bu' :'ブ', 'be' :'ベ', 'bo' :'ボ',
        'pa' :'パ', 'pi' :'ピ', 'pu' :'プ', 'pe' :'ペ', 'po' :'ポ',
        
        'kya':'キャ', 'kyi':'キィ', 'kyu':'キュ', 'kye':'キェ', 'kyo':'キョ',
        'gya':'ギャ', 'gyi':'ギィ', 'gyu':'ギュ', 'gye':'ギェ', 'gyo':'ギョ',
        'sha':'シャ',               'shu':'シュ', 'she':'シェ', 'sho':'ショ',
        'ja' :'ジャ',               'ju' :'ジュ', 'je' :'ジェ', 'jo' :'ジョ',
        'cha':'チャ',               'chu':'チュ', 'che':'チェ', 'cho':'チョ',
        'dya':'ヂャ', 'dyi':'ヂィ', 'dyu':'ヂュ', 'dhe':'デェ', 'dyo':'ヂョ',
        'nya':'ニャ', 'nyi':'ニィ', 'nyu':'ニュ', 'nye':'ニェ', 'nyo':'ニョ',
        'hya':'ヒャ', 'hyi':'ヒィ', 'hyu':'ヒュ', 'hye':'ヒェ', 'hyo':'ヒョ',
        'bya':'ビャ', 'byi':'ビィ', 'byu':'ビュ', 'bye':'ビェ', 'byo':'ビョ',
        'pya':'ピャ', 'pyi':'ピィ', 'pyu':'ピュ', 'pye':'ピェ', 'pyo':'ピョ',
        'mya':'ミャ', 'myi':'ミィ', 'myu':'ミュ', 'mye':'ミェ', 'myo':'ミョ',
        'rya':'リャ', 'ryi':'リィ', 'ryu':'リュ', 'rye':'リェ', 'ryo':'リョ',
        # 'fa' :'ファ', 'fi' :'フィ',               'fe' :'フェ', 'fo' :'フォ',
        # 'wi' :'ウィ', 'we' :'ウェ', 
        # 'va' :'ヴァ', 'vi' :'ヴィ', 've' :'ヴェ', 'vo' :'ヴォ',
        
        # 'kwa':'クァ', 'kwi':'クィ', 'kwu':'クゥ', 'kwe':'クェ', 'kwo':'クォ',
        # 'kha':'クァ', 'khi':'クィ', 'khu':'クゥ', 'khe':'クェ', 'kho':'クォ',
        # 'gwa':'グァ', 'gwi':'グィ', 'gwu':'グゥ', 'gwe':'グェ', 'gwo':'グォ',
        # 'gha':'グァ', 'ghi':'グィ', 'ghu':'グゥ', 'ghe':'グェ', 'gho':'グォ',
        # 'swa':'スァ', 'swi':'スィ', 'swu':'スゥ', 'swe':'スェ', 'swo':'スォ',
        # 'swa':'スァ', 'swi':'スィ', 'swu':'スゥ', 'swe':'スェ', 'swo':'スォ',
        # 'zwa':'ズヮ', 'zwi':'ズィ', 'zwu':'ズゥ', 'zwe':'ズェ', 'zwo':'ズォ',
        # 'twa':'トァ', 'twi':'トィ', 'twu':'トゥ', 'twe':'トェ', 'two':'トォ',
        # 'dwa':'ドァ', 'dwi':'ドィ', 'dwu':'ドゥ', 'dwe':'ドェ', 'dwo':'ドォ',
        # 'mwa':'ムヮ', 'mwi':'ムィ', 'mwu':'ムゥ', 'mwe':'ムェ', 'mwo':'ムォ',
        # 'bwa':'ビヮ', 'bwi':'ビィ', 'bwu':'ビゥ', 'bwe':'ビェ', 'bwo':'ビォ',
        # 'pwa':'プヮ', 'pwi':'プィ', 'pwu':'プゥ', 'pwe':'プェ', 'pwo':'プォ',
        # 'phi':'プィ', 'phu':'プゥ', 'phe':'プェ', 'pho':'フォ',
    }
    
    
    romaji_asist = {
        # 'si' :'シ'  , 'ti' :'チ'  , 'hu' :'フ' , 'zi':'ジ',
        # 'sya':'シャ', 'syu':'シュ', 'syo':'ショ',
        # 'tya':'チャ', 'tyu':'チュ', 'tyo':'チョ',
        # 'cya':'チャ', 'cyu':'チュ', 'cyo':'チョ',
        # 'jya':'ジャ', 'jyu':'ジュ', 'jyo':'ジョ', 'pha':'ファ', 
        # 'qa' :'クァ', 'qi' :'クィ', 'qu' :'クゥ', 'qe' :'クェ', 'qo':'クォ',
        
        # 'ca' :'カ', 'ci':'シ', 'cu':'ク', 'ce':'セ', 'co':'コ',
        # 'la' :'ラ', 'li':'ィ', 'lu':'ル', 'le':'レ', 'lo':'ロ',

        # 'mb' :'ム', 'py':'パイ', 'tho': 'ソ', 'thy':'ティ', 'oh':'オウ',
        # 'by':'ビィ', 'cy':'シィ', 'dy':'ディ', 'fy':'フィ', 'gy':'ジィ',
        # 'hy':'シー', 'ly':'リィ', 'ny':'ニィ', 'my':'ミィ', 'ry':'リィ',
        # 'ty':'ティ', 'vy':'ヴィ', 'zy':'ジィ',
        
        # 'li':'ィ',
        # 'b':'ブ', 'c':'ク', 'd':'ド', 'f':'フ'  , 'g':'グ', 'h':'フ', 'j':'ジ',
        # 'k':'ク', 'l':'ル', 'm':'ム', 'p':'プ'  , 'q':'ク', 'r':'ル', 's':'ス',
        # 't':'ト', 'v':'ヴ', 'w':'ゥ', 'x':'クス', 'y':'ィ', 'z':'ズ',
        }
    

    kana_asist = { 'la':'ァ', 'li':'ィ', 'lu':'ゥ', 'le':'ェ', 'lo':'ォ', }
    
    
    def __romaji2kana():
        romaji_dict = {}
        for tbl in master, romaji_asist:
            for k, v in tbl.items(): romaji_dict[k] = v
        
        romaji_keys = list(romaji_dict.keys())
        romaji_keys.sort(key=lambda x:len(x), reverse=True)
        
        re_roma2kana = re.compile("|".join(map(re.escape, romaji_keys)))
        # m の後ろにバ行、パ行のときは "ン" と変換
        rx_mba = re.compile("m(b|p)([aiueo])")
        # 子音が続く時は "ッ" と変換
        rx_xtu = re.compile(r"([bcdfghjklmpqrstvwxyz])\1")
        # 母音が続く時は "ー" と変換
        rx_a__ = re.compile(r"([aiueo])\1")
        
        def _romaji2katakana(text):
            result = text.lower()
            result = rx_mba.sub(r"ン\1\2", result)
            result = rx_xtu.sub(r"ッ\1"  , result)
            result = rx_a__.sub(r"\1ー"  , result)
            return re_roma2kana.sub(lambda x: romaji_dict[x.group(0)], result)
        
        def _romaji2hiragana(text):
            result = _romaji2katakana(text)
            return katakana2hiragana(result)
        
        return _romaji2katakana, _romaji2hiragana
    
    
    def __kana2romaji():
        kana_dict = {}
        for tbl in master, kana_asist:
            for k, v in tbl.items(): kana_dict[v] = k

        kana_keys = list(kana_dict.keys())
        kana_keys.sort(key=lambda x:len(x), reverse=True)
        
        re_kana2roma = re.compile("|".join(map(re.escape, kana_keys)))
        rx_xtu = re.compile("ッ(.)") # 小さい "ッ" は直後の文字を２回に変換
        # rx_ltu = re.compile("ッ$"  ) # 最後の小さい "ッ" は消去(?)
        # rx_er  = re.compile("(.)ー") # "ー"は直前の文字を２回に変換
        # rx_n   = re.compile(r"n(b|p)([aiueo])") # n の後ろが バ行、パ行 なら m に修正
        # rx_oo  = re.compile(r"([aiueo])\1")      # oosaka → osaka
        
        def _kana2romaji(text):
            result = hiragana2katakana(text)
            result = re_kana2roma.sub(lambda x: kana_dict[x.group(0)], result)
            result = rx_xtu.sub(r"\1\1" , result)
            # result = rx_ltu.sub(r""     , result)
            # result = rx_er.sub (r"\1\1" , result)
            # result = rx_n.sub  (r"m\1\2", result)
            # result = rx_oo.sub (r"\1"   , result)
            return result
        return _kana2romaji
    
    a, b = __romaji2kana()
    c    = __kana2romaji()
    
    return  a, b, c

romaji2katakana, romaji2hiragana, kana2romaji = _make_romaji_convertor()
