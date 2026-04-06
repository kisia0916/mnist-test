import csv
import tkinter as tk
from pathlib import Path

import numpy as np


GRID_SIZE = 8
CELL_SIZE = 56
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
DATASET_FILE = Path("digits_dataset_drawn.csv")


class DigitDatasetTool:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("MNIST 8x8 Data Creator")
        self.master.resizable(False, False)

        self.values = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.rect_ids = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        self._build_ui()
        self._draw_grid()
        self._refresh_canvas()

    def _build_ui(self) -> None:
        root = tk.Frame(self.master, padx=12, pady=12)
        root.pack()

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.canvas.grid(row=0, column=0, columnspan=4)

        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<Button-1>", self._paint)
        self.canvas.bind("<B3-Motion>", self._erase)
        self.canvas.bind("<Button-3>", self._erase)

        tk.Label(root, text="ラベル (0-9)").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.label_entry = tk.Entry(root, width=8)
        self.label_entry.insert(0, "0")
        self.label_entry.grid(row=1, column=1, sticky="w", pady=(10, 0))

        tk.Button(root, text="保存", width=12, command=self.save_sample).grid(row=1, column=2, padx=4, pady=(10, 0))
        tk.Button(root, text="クリア", width=12, command=self.clear_grid).grid(row=1, column=3, padx=4, pady=(10, 0))

        self.status_var = tk.StringVar(value="左ドラッグ: 描画 / 右ドラッグ: 消しゴム")
        tk.Label(root, textvariable=self.status_var, anchor="w", fg="#333333").grid(
            row=2,
            column=0,
            columnspan=4,
            sticky="we",
            pady=(10, 0),
        )

        tk.Button(root, text="64要素配列を表示", width=20, command=self.show_flattened_array).grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="we",
            pady=(8, 0),
        )

    def _draw_grid(self) -> None:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x0 = col * CELL_SIZE
                y0 = row * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                rect_id = self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill="white",
                    outline="#cccccc",
                    width=1,
                )
                self.rect_ids[row][col] = rect_id

    def _event_to_cell(self, event: tk.Event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return int(row), int(col)
        return None, None

    def _paint(self, event: tk.Event) -> None:
        row, col = self._event_to_cell(event)
        if row is None or col is None:
            return

        # 中心は濃く、周囲は薄く塗ることでドラッグ時の線をなめらかにする。
        for dr, dc, value in [
            (0, 0, 1.0),
            (-1, 0, 0.25),
            (1, 0, 0.25),
            (0, -1, 0.25),
            (0, 1, 0.25),
        ]:
            rr = row + dr
            cc = col + dc
            if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE:
                self.values[rr, cc] = max(float(self.values[rr, cc]), value)

        self._refresh_canvas()

    def _erase(self, event: tk.Event) -> None:
        row, col = self._event_to_cell(event)
        if row is None or col is None:
            return
        self.values[row, col] = 0.0
        self._refresh_canvas()

    def _refresh_canvas(self) -> None:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                v = float(self.values[row, col])
                gray = int(255 * (1.0 - v))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                self.canvas.itemconfig(self.rect_ids[row][col], fill=color)

    def clear_grid(self) -> None:
        self.values.fill(0.0)
        self._refresh_canvas()
        self.status_var.set("グリッドをクリアしました")

    def _get_label(self):
        text = self.label_entry.get().strip()
        if not text.isdigit():
            self.status_var.set("ラベルは 0-9 の整数で入力してください")
            return None
        label = int(text)
        if not 0 <= label <= 9:
            self.status_var.set("ラベルは 0-9 の範囲で入力してください")
            return None
        return label

    def to_column_flattened(self) -> list[float]:
        # 行優先で結合: row0の左→右, row1の左→右, ... の順で64要素にする。
        return self.values.flatten().astype(float).tolist()

    def show_flattened_array(self) -> None:
        arr = self.to_column_flattened()
        self.status_var.set(f"配列長: {len(arr)} / 先頭8要素: {arr[:8]}")
        print(arr)

    def save_sample(self) -> None:
        label = self._get_label()
        if label is None:
            return

        arr64 = self.to_column_flattened()
        row = arr64 + [label]

        with DATASET_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.status_var.set(f"保存完了: {DATASET_FILE.name} に 1件追加 (label={label})")


def main() -> None:
    root = tk.Tk()
    DigitDatasetTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
