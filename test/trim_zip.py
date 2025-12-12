import zipfile
import io
from pathlib import Path

import pandas as pd


def trim_zip_file(input_zip_path,
                  output_zip_path=None,
                  cut_last_sec=5.0,
                  keep_before_sec=30.0):
    """
    input_zip_path:  원본 zip 경로 (예: 'x.zip')
    output_zip_path: 결과 zip 경로 (None이면 '_trimmed' 붙여서 자동생성)
    cut_last_sec:   맨 마지막에서 잘라낼 시간 (초)
    keep_before_sec:그 앞쪽에서 남길 시간 (초) → 기본 30초
    """
    input_zip_path = Path(input_zip_path)

    if output_zip_path is None:
        output_zip_path = input_zip_path.with_name(
            input_zip_path.stem + "_trimmed" + input_zip_path.suffix
        )

    with zipfile.ZipFile(input_zip_path, "r") as zin, \
            zipfile.ZipFile(output_zip_path, "w",
                            compression=zipfile.ZIP_DEFLATED) as zout:

        for info in zin.infolist():
            name = info.filename

            with zin.open(name) as f:
                raw_data = f.read()

            # CSV 파일만 처리
            if name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(io.BytesIO(raw_data))
                except Exception:
                    # 혹시 읽기 실패하면 그냥 원본 복사
                    zout.writestr(name, raw_data)
                    continue

                # 시간 정보가 있는 CSV만 자르기
                if "seconds_elapsed" in df.columns:
                    t_min = df["seconds_elapsed"].min()
                    t_max = df["seconds_elapsed"].max()

                    # 남길 구간 계산:
                    # 1) 마지막 cut_last_sec(기본 5초)는 버림
                    t_end_keep = t_max - cut_last_sec        # 남길 구간의 끝
                    # 2) 그 앞쪽 keep_before_sec(기본 30초)만 남김
                    t_start_keep = t_end_keep - keep_before_sec  # 남길 구간의 시작

                    # 전체 길이가 30+5초보다 짧으면,
                    # "마지막 5초만 버리고 나머지 전체"를 남기기 위해
                    if t_start_keep < t_min:
                        t_start_keep = t_min

                    mask = (
                        (df["seconds_elapsed"] >= t_start_keep) &
                        (df["seconds_elapsed"] <= t_end_keep)
                    )
                    trimmed = df.loc[mask].copy()

                    # CSV로 다시 기록
                    buf = io.StringIO()
                    trimmed.to_csv(buf, index=False)
                    zout.writestr(name, buf.getvalue())
                else:
                    # seconds_elapsed 없으면 그대로 복사
                    zout.writestr(name, raw_data)
            else:
                # CSV가 아닌 나머지 파일(Metadata 등)은 그대로 복사
                zout.writestr(name, raw_data)

    print(f"완료: {input_zip_path.name} -> {Path(output_zip_path).name}")


if __name__ == "__main__":
    # 예시 사용법
    # 같은 폴더에 있는 모든 zip 처리
    for zip_path in Path(".").glob("*.zip"):
        trim_zip_file(zip_path)

