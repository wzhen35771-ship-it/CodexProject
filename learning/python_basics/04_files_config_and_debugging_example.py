"""用标准库练习路径、JSON、命令行与异常；全部写入临时目录。"""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import traceback


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description="检查解释器、读写临时配置并观察异常")
    parser.add_argument("--batch-size", type=positive_integer, default=4)
    parser.add_argument(
        "--show-traceback", action="store_true", help="显示一个被捕获的 JSON 错误堆栈"
    )
    arguments = parser.parse_args()

    script_path = Path(__file__).resolve()
    print("当前解释器：", sys.executable)
    print("当前工作目录：", Path.cwd())
    print("脚本所在目录：", script_path.parent)
    print("命令行 batch_size：", arguments.batch_size)

    config = {"batch_size": arguments.batch_size, "epochs": 2, "relation_names": ["左侧", "右侧", "上方"]}
    with tempfile.TemporaryDirectory(prefix="python_learning_") as temporary_directory:
        config_path = Path(temporary_directory) / "config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
        if loaded_config != config:
            raise RuntimeError("配置保存和读取的结果不一致")
        print("读回的配置：", loaded_config)
        print("第一类关系：", loaded_config["relation_names"][0])
        try:
            json.loads('{"batch_size": }')
        except json.JSONDecodeError as error:
            print(f"捕获预期错误：{type(error).__name__}，第 {error.lineno} 行第 {error.colno} 列")
            if arguments.show_traceback:
                traceback.print_exc(file=sys.stdout)
        else:
            raise RuntimeError("错误 JSON 应当触发 JSONDecodeError")

    print("离开 with 后，临时目录已清理：", not config_path.parent.exists())
    print("路径、配置与异常示例完成；未安装任何依赖。")


if __name__ == "__main__":
    main()
