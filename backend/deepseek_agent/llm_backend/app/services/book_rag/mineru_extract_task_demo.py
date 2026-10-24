from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://mineru.net/api/v4"
DEFAULT_CREATE_URL = f"{DEFAULT_BASE_URL}/extract/task"
DEFAULT_MODEL_VERSION = "vlm"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("mineru_results")
FINAL_STATES = {"done", "failed"}


def build_headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }


def create_extract_task(
    token: str,
    file_url: str,
    model_version: str = DEFAULT_MODEL_VERSION,
    is_ocr: bool = True,
    enable_formula: bool = True,
    enable_table: bool = True,
    language: str = "ch",
    page_ranges: str | None = None,
    data_id: str | None = None,
    no_cache: bool = False,
    cache_tolerance: int = 900,
    extra_formats: list[str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": file_url,
        "model_version": model_version,
        "is_ocr": is_ocr,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
        "language": language,
        "no_cache": no_cache,
        "cache_tolerance": cache_tolerance,
    }

    if page_ranges:
        payload["page_ranges"] = page_ranges
    if data_id:
        payload["data_id"] = data_id
    if extra_formats:
        payload["extra_formats"] = extra_formats

    response = requests.post(
        DEFAULT_CREATE_URL,
        headers=build_headers(token),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_task_result(token: str, task_id: str, timeout: int = 60) -> dict[str, Any]:
    query_url = f"{DEFAULT_CREATE_URL}/{task_id}"
    response = requests.get(query_url, headers=build_headers(token), timeout=timeout)
    response.raise_for_status()
    return response.json()


def wait_for_task(
    token: str,
    task_id: str,
    poll_interval: int = 5,
    max_wait_seconds: int = 900,
    timeout: int = 60,
) -> dict[str, Any]:
    start_time = time.time()

    while True:
        result = get_task_result(token=token, task_id=task_id, timeout=timeout)
        data = result.get("data", {})
        state = data.get("state", "")

        print(f"当前状态: {state or 'unknown'}")
        if state == "running":
            progress = data.get("extract_progress") or {}
            extracted_pages = progress.get("extracted_pages")
            total_pages = progress.get("total_pages")
            start_at = progress.get("start_time")
            print(
                f"解析进度: {extracted_pages}/{total_pages}, start_time={start_at}"
            )
        elif state == "failed":
            print(f"解析失败: {data.get('err_msg', '')}")
        elif state == "done":
            print("解析完成")

        if state in FINAL_STATES:
            return result

        elapsed = time.time() - start_time
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"等待任务完成超时，task_id={task_id}")

        time.sleep(poll_interval)


def download_file(file_url: str, output_path: Path, timeout: int = 120) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(file_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MinerU 官方 extract/task 接口示例（含轮询，可开启 OCR）。")
    parser.add_argument(
        "--token",
        default=os.getenv("MINERU_API_TOKEN", ""),
        help="MinerU API token，也可通过环境变量 MINERU_API_TOKEN 提供。",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="待解析文档的公网可访问 URL。",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help="模型版本：pipeline / vlm / MinerU-HTML，默认 vlm。",
    )
    parser.add_argument(
        "--language",
        default="ch",
        help="文档语言，默认 ch。",
    )
    parser.add_argument(
        "--page-ranges",
        default="",
        help="页码范围，例如 1-20 或 2,4-6。为空表示全量解析。",
    )
    parser.add_argument(
        "--data-id",
        default="",
        help="业务数据唯一标识，可选。",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="轮询间隔秒数，默认 5。",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=900,
        help="最大等待时长（秒），默认 900。",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="单次 HTTP 请求超时时间（秒），默认 60。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="结果输出目录。",
    )
    parser.add_argument(
        "--extra-formats",
        nargs="*",
        default=[],
        choices=["docx", "html", "latex"],
        help="额外导出格式，可选 docx html latex。",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="默认开启 OCR；传入此参数可关闭 OCR。",
    )
    parser.add_argument(
        "--disable-formula",
        action="store_true",
        help="默认开启公式识别；传入此参数可关闭。",
    )
    parser.add_argument(
        "--disable-table",
        action="store_true",
        help="默认开启表格识别；传入此参数可关闭。",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="绕过缓存，从 URL 获取最新内容。",
    )
    parser.add_argument(
        "--cache-tolerance",
        type=int,
        default=900,
        help="缓存容忍时间（秒），默认 900。",
    )
    parser.add_argument(
        "--download-zip",
        action="store_true",
        help="任务完成后自动下载 full_zip_url 对应压缩包。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token.strip()
    if not token:
        raise SystemExit("缺少 token，请通过 --token 或环境变量 MINERU_API_TOKEN 提供。")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_result = create_extract_task(
        token=token,
        file_url=args.url,
        model_version=args.model_version,
        is_ocr=not args.disable_ocr,
        enable_formula=not args.disable_formula,
        enable_table=not args.disable_table,
        language=args.language,
        page_ranges=args.page_ranges.strip() or None,
        data_id=args.data_id.strip() or None,
        no_cache=args.no_cache,
        cache_tolerance=args.cache_tolerance,
        extra_formats=args.extra_formats or None,
        timeout=args.timeout,
    )

    create_result_path = output_dir / "create_task_response.json"
    create_result_path.write_text(
        json.dumps(create_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("创建任务响应:")
    print(json.dumps(create_result, ensure_ascii=False, indent=2))
    print(f"创建任务结果已保存到: {create_result_path.resolve()}")

    task_id = (create_result.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError("未从创建任务响应中拿到 task_id，请检查返回结果。")

    print(f"\n任务已创建，task_id={task_id}")
    final_result = wait_for_task(
        token=token,
        task_id=task_id,
        poll_interval=args.poll_interval,
        max_wait_seconds=args.max_wait_seconds,
        timeout=args.timeout,
    )

    final_result_path = output_dir / "final_task_result.json"
    final_result_path.write_text(
        json.dumps(final_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"最终任务结果已保存到: {final_result_path.resolve()}")

    final_data = final_result.get("data") or {}
    if final_data.get("state") == "done":
        full_zip_url = final_data.get("full_zip_url", "")
        if full_zip_url:
            print(f"\nfull_zip_url: {full_zip_url}")
            if args.download_zip:
                zip_path = output_dir / f"{task_id}.zip"
                download_file(full_zip_url, zip_path)
                print(f"压缩包已下载到: {zip_path.resolve()}")
    elif final_data.get("state") == "failed":
        raise RuntimeError(f"解析失败: {final_data.get('err_msg', '')}")


if __name__ == "__main__":
    main()
