# -*- coding: utf-8 -*-
import argparse
import time
import os

from mmd.utils.MLogger import MLogger

logger = MLogger(__name__)


def show_worked_time(elapsed_time):
    # 経過秒数を時分秒に変換
    td_m, td_s = divmod(elapsed_time, 60)
    td_h, td_m = divmod(td_m, 60)

    if td_m == 0:
        worked_time = "{0:02d}秒".format(int(td_s))
    elif td_h == 0:
        worked_time = "{0:02d}分{1:02d}秒".format(int(td_m), int(td_s))
    else:
        worked_time = "{0:02d}時間{1:02d}分{2:02d}秒".format(int(td_h), int(td_m), int(td_s))

    return worked_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-dir', type=str, dest='parent_dir', default='', help='Process parent dir path')
    parser.add_argument('--process', type=str, dest='process', default='', help='Process to be executed')
    parser.add_argument('--audio-dir', type=str, dest='audio_dir', default='', help='Prepared image directory')
    parser.add_argument('--audio-file', type=str, dest='audio_file', default='', help='Audio file path')
    parser.add_argument('--lyrics-file', type=str, dest='lyrics_file', default='', help='Audio file path')
    parser.add_argument('--threshold', type=int, dest='threshold', default=0.2, help='threshold')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='Log level')
    parser.add_argument("--log-mode", type=int, dest='log_mode', default=0, help='Log output mode')

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode)
    result = True

    start = time.time()

    logger.info("MMD自動トレース（リップ）開始\n　処理対象映像ファイル: {0}\n　処理内容: {1}", args.audio_file, args.process, decoration=MLogger.DECORATION_BOX)

    if result and "vocals" in args.process:
        # リップモーション生成
        import mmd.vocals
        result, args.audio_dir = mmd.vocals.execute(args)

    if result and "lip" in args.process:
        # リップモーション生成
        import mmd.lip
        result = mmd.lip.execute(args)

    elapsed_time = time.time() - start

    logger.info("MMD自動トレース（リップ）終了\n　処理対象映像ファイル: {0}\n　処理内容: {1}\n　トレース結果: {2}\n　処理時間: {3}", \
                args.audio_file, args.process, args.audio_dir, show_worked_time(elapsed_time), decoration=MLogger.DECORATION_BOX)

    # 終了音を鳴らす
    if os.name == "nt":
        # Windows
        try:
            import winsound
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
        except Exception:
            pass

