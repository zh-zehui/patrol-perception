#!/usr/bin/env python3
"""Rebuild native SVG diagrams and a chart from committed historical metrics."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"
BG = "#0b1425"
CARD = "#142238"
TEXT = "#e8f0fb"
MUTED = "#9db0cb"
TEAL = "#54d6bd"
BLUE = "#73a9ff"
AMBER = "#f5c56e"


class SVG:
    def __init__(self, height: int, title: str, description: str):
        self.height = height
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="{height}" viewBox="0 0 1440 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#607b9f"/></marker></defs>',
                      f'<rect width="1440" height="{height}" rx="24" fill="{BG}"/>',
                      '<g font-family="Inter, Noto Sans CJK SC, Microsoft YaHei, sans-serif">']

    def text(self, x: int, y: int, text: str, size: int = 20, color: str = TEXT, bold: bool = False):
        self.items.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(text)}</text>')

    def rect(self, x, y, w, h, fill=CARD, stroke="#263a55"):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{stroke}"/>')

    def line(self, x1, y1, x2, y2):
        self.items.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#607b9f" stroke-width="2" marker-end="url(#arrow)"/>')

    def card(self, x, y, w, title, subtitle, detail, accent=TEAL):
        self.rect(x, y, w, 126)
        self.items.append(f'<rect x="{x+20}" y="{y+22}" width="5" height="22" rx="2" fill="{accent}"/>')
        self.text(x+36, y+40, title, 25, bold=True)
        self.text(x+24, y+77, subtitle, 18, MUTED)
        self.text(x+24, y+106, detail, 16, accent)

    def save(self, name):
        self.items += ['</g>', '</svg>']
        (OUT / name).write_text("\n".join(self.items) + "\n")


def architecture():
    s = SVG(790, "巡逻视觉系统架构", "已实现的数据标注训练闭环和离线检测管线；ROS2 为规划项")
    s.text(52, 52, "PATROL PERCEPTION / SYSTEM OVERVIEW", 17, TEAL, True)
    s.text(52, 109, "从数据审核，到离线感知", 42, bold=True)
    s.text(52, 148, "浏览器标注 · 独立模型服务 · 可追溯评测 · 结构化事件输出", 22, MUTED)
    s.text(52, 205, "01  DATA &amp; MODEL LOOP".replace("&amp;", "&"), 16, BLUE, True)
    cards = [
        (52, "数据来源与整理", "公开数据 / 自采数据 / 类别映射", "datasets/scripts/"),
        (398, "模型候选 + 人工审核", "视觉 API 或 SAM 3 → 草稿 → 提交", "annotation_tool/"),
        (744, "数据构建与训练", "多源合并 / 抽样 / YOLO 训练", "datasets.py + train scripts"),
        (1090, "评测与记录", "每类 AP / 重点类 / 基线差值", "eval/ + metrics JSON"),
    ]
    for x, title, subtitle, detail in cards:
        s.card(x, 228, 298, title, subtitle, detail, BLUE)
    for x in (350, 696, 1042):
        s.line(x+5, 291, x+42, 291)
    s.text(52, 410, "02  OFFLINE INFERENCE", 16, TEAL, True)
    for x, title, subtitle, detail in [
        (52, "有序图片序列", "单张图片或文件名排序的目录", "输入图像"),
        (398, "YOLO + IoU 跟踪", "类别 / 置信度 / 检测框 / Track ID", "models/ + track/"),
        (744, "基础事件规则", "目标发现 / 人员聚集 / 夜间逗留", "rules/"),
        (1090, "JSON 结果", "逐帧 FrameResult / 事件汇总", "pipeline/"),
    ]:
        s.card(x, 433, 298, title, subtitle, detail)
    for x in (350, 696, 1042):
        s.line(x+5, 496, x+42, 496)
    s.rect(52, 610, 1336, 118, "#101e33")
    s.text(78, 648, "运行边界", 22, bold=True)
    s.text(78, 682, "轻量标注环境与 GPU 服务隔离；数据、权重、密钥留在本地。", 21, MUTED)
    s.text(78, 710, "ROS2 桥接与实时车端集成：规划中。图示为静态架构，不是实时运行监控。", 18, AMBER)
    s.save("architecture.svg")


def status():
    s = SVG(660, "系统模块状态", "五个模块具备代码实现；ROS2 集成为规划项；状态不等于部署验收")
    s.text(52, 53, "ENGINEERING STATUS / STATIC SNAPSHOT", 17, TEAL, True)
    s.text(52, 108, "系统能力与实现边界", 38, bold=True)
    s.text(52, 146, "按仓库代码核对 · 模型服务与现场表现需单独验证", 22, MUTED)
    entries = [
        ("数据工程", "转换 · 类别映射 · 合并 · 统计", "已有实现", TEAL),
        ("标注与审核", "门户 · 草稿 · 历史 · 批量提交", "已有实现", TEAL),
        ("辅助标注适配", "视觉 API / SAM 3 独立服务", "服务需配置", BLUE),
        ("离线感知", "图片检测 · IoU 跟踪 · 基础规则", "已有实现", TEAL),
        ("检测评测", "按类指标 · 基线比较 · 报告", "已有历史记录", BLUE),
        ("ROS2 与实时集成", "桥接节点 · 车端连续运行验证", "规划中", AMBER),
    ]
    for i, (title, subtitle, state, accent) in enumerate(entries):
        x = 52 + (i % 3) * 452
        y = 190 + (i // 3) * 190
        s.rect(x, y, 432, 166)
        s.text(x+24, y+43, title, 25, bold=True)
        s.text(x+24, y+80, subtitle, 19, MUTED)
        s.items.append(f'<circle cx="{x+30}" cy="{y+125}" r="5" fill="{accent}"/>')
        s.text(x+46, y+132, state, 20, accent)
    s.text(52, 610, "事件目录中的 ship / partial 是配置声明；当前高级多模态事件未完整接入离线管线。", 19, MUTED)
    s.save("system-status.svg")


def workflow():
    s = SVG(480, "标注审核状态流转", "候选复制为草稿，人工标记可提交后写入正式标签，旧标签保留历史")
    s.text(52, 53, "ANNOTATION REVIEW / STATE FLOW", 17, TEAL, True)
    s.text(52, 105, "候选、草稿与正式标签分层", 36, bold=True)
    for x, title, subtitle, detail, accent in [
        (52, "模型候选", "不修改正式标签", "candidates/", BLUE),
        (398, "人工编辑草稿", "复制候选或正式基线后编辑", "drafts/", BLUE),
        (744, "标记可提交", "人工确认后进入批量提交范围", "draft_ready", AMBER),
        (1090, "提交正式标签", "更新 YOLO TXT 与审核状态", "labels/ · reviewed", TEAL),
    ]:
        s.card(x, 158, 298, title, subtitle, detail, accent)
    for x in (350, 696, 1042):
        s.line(x+5, 221, x+42, 221)
    s.rect(398, 326, 298, 78)
    s.text(422, 358, "需复核 → 返回人工编辑", 20, AMBER)
    s.text(422, 386, "needs_review", 17, MUTED)
    s.line(547, 325, 547, 290)
    s.rect(744, 326, 644, 78)
    s.text(768, 358, "提交前备份旧标签；提交成功追加操作记录", 21, TEAL)
    s.text(768, 387, "history/ + commit_log.jsonl", 18, MUTED)
    s.line(1239, 289, 1239, 320)
    s.text(52, 450, "模型批量标注只生成候选；批量提交只处理人工明确标记为 draft_ready 的图片。", 19, MUTED)
    s.save("review-workflow.svg")


def evaluation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    base = ROOT / "docs/benchmarks/w2_joint"
    val = json.loads((base / "metrics_val.json").read_text())
    test = json.loads((base / "metrics_test.json").read_text())
    names = val["names"]
    fig, ax = plt.subplots(figsize=(14, 9.5), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(names))
    a = ax.barh(y-.19, [val["per_class"][n]["ap50"] for n in names], height=.32, color=BLUE, label="Validation")
    b = ax.barh(y+.19, [test["per_class"][n]["ap50"] for n in names], height=.32, color=TEAL, label="Test")
    ax.bar_label(a, fmt="%.3f", padding=5, color=BLUE, fontsize=9)
    ax.bar_label(b, fmt="%.3f", padding=5, color=TEAL, fontsize=9)
    ax.set_yticks(y, names, color=TEXT, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.07)
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlabel("AP@50  |  higher is better", color=MUTED, labelpad=14)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#243650", linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower right", facecolor=CARD, edgecolor="#263a55", labelcolor=TEXT, fontsize=11)
    fig.text(.05, .95, "W2 JOINT / HISTORICAL DETECTION EVALUATION", color=TEAL, fontsize=12, weight="bold")
    fig.text(.05, .902, "14 classes. Two splits. Recorded results.", color=TEXT, fontsize=25, weight="bold")
    fig.text(.05, .863, "2026-08-18  |  imgsz 640  |  mAP@50  val 0.6571 / test 0.6544", color=MUTED, fontsize=12)
    fig.text(.05, .03, "Source: docs/benchmarks/w2_joint/metrics_{val,test}.json  |  Not rerun for this showcase; no FPS claim.", color=MUTED, fontsize=10)
    fig.subplots_adjust(left=.23, right=.94, top=.81, bottom=.12)
    fig.savefig(OUT / "evaluation.png", dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    status()
    workflow()
    evaluation()
    print("Rebuilt architecture, module status, review flow and historical AP chart.")


if __name__ == "__main__":
    main()
