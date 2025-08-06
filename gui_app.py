import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# 您封装的处理函数
from face_processor import process_and_draw_face_geometry


# ==================  CollapsibleFrame  ==================
class CollapsibleFrame(tk.Frame):
    """可折叠容器，点击标题栏展开/收起内容"""
    def __init__(self, parent, text="", *args, **kw):
        super().__init__(parent, *args, **kw)
        self.columnconfigure(0, weight=1)
        self.show = tk.IntVar(value=0)          # 0=收起
        # 标题栏
        self.title = tk.Button(
            self, text=text + "  ▼", bd=1, relief="raised",
            anchor="w", command=self.toggle)
        self.title.grid(row=0, column=0, sticky="ew")
        # 内容区
        self.sub = tk.Frame(self, bd=2, relief="sunken")
        self.sub.grid(row=1, column=0, sticky="ew")
        self.sub.grid_remove()                  # 默认隐藏

    def toggle(self):
        if self.show.get():
            self.sub.grid_remove()
            self.title.config(text=self.title["text"].replace("▲", "▼"))
        else:
            self.sub.grid()
            self.title.config(text=self.title["text"].replace("▼", "▲"))
        self.show.set(not self.show.get())
# =========================================================


class FaceGeometryApp:
    def __init__(self, master):
        self.master = master
        master.title("人脸几何特征分析工具")
        master.geometry("1600x820")             # 初始高度减小
        master.resizable(True, True)

        self.image_path = None
        self.original_cv2_image = None
        self.processed_cv2_image = None
        self.input_image_tk = None
        self.output_image_tk = None

        # 默认肤色检测参数（完整）
        self.default_hsv_lower_full = [5, 53, 92]
        self.default_hsv_upper_full = [18, 140, 225]
        self.default_ycbcr_lower_full = [95, 135, 80]
        self.default_ycbcr_upper_full = [220, 165, 116]

        # --- 顶部控制区 ---
        self.frame_controls = tk.Frame(master, padx=10, pady=10)
        self.frame_controls.pack(side=tk.TOP, fill=tk.X)

        tk.Label(self.frame_controls, text="人脸几何特征分析", font=("Helvetica", 16, "bold")).pack(pady=5)

        self.btn_select_image = tk.Button(self.frame_controls, text="选择图片", command=self.select_image)
        self.btn_select_image.pack(side=tk.LEFT, padx=5)

        self.label_image_path = tk.Label(self.frame_controls, text="未选择图片", width=60, anchor="w")
        self.label_image_path.pack(side=tk.LEFT, padx=5)

        self.btn_process_image = tk.Button(self.frame_controls, text="处理图片", command=self.process_image, state=tk.DISABLED)
        self.btn_process_image.pack(side=tk.LEFT, padx=5)

        self.btn_save_image = tk.Button(self.frame_controls, text="保存图片", command=self.save_image, state=tk.DISABLED)
        self.btn_save_image.pack(side=tk.LEFT, padx=5)

        self.label_status = tk.Label(self.frame_controls, text="", fg="blue")
        self.label_status.pack(side=tk.LEFT, padx=10)

        # --- 折叠式参数调节框 ---
        self.collapse = CollapsibleFrame(master, text="肤色检测参数调节")
        self.collapse.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.frame_sliders = self.collapse.sub     # 向后兼容

        self.scales = {}
        self.param_configs = {
            "hsv_V": {
                "lower": {"default": self.default_hsv_lower_full[2], "range": (0, 255),
                          "label": "HSV 亮度(V) 下限:", "full_idx": 2, "full_list_type": "hsv_lower"},
                "upper": {"default": self.default_hsv_upper_full[2], "range": (0, 255),
                          "label": "HSV 亮度(V) 上限:", "full_idx": 2, "full_list_type": "hsv_upper"},
            },
            "ycbcr_Y": {
                "lower": {"default": self.default_ycbcr_lower_full[0], "range": (0, 255),
                          "label": "YCbCr 亮度(Y) 下限:", "full_idx": 0, "full_list_type": "ycbcr_lower"},
                "upper": {"default": self.default_ycbcr_upper_full[0], "range": (0, 255),
                          "label": "YCbCr 亮度(Y) 上限:", "full_idx": 0, "full_list_type": "ycbcr_upper"},
            },
            "ycbcr_Cb": {
                "lower": {"default": self.default_ycbcr_lower_full[1], "range": (0, 255),
                          "label": "YCbCr 色度(Cb) 下限:", "full_idx": 1, "full_list_type": "ycbcr_lower"},
                "upper": {"default": self.default_ycbcr_upper_full[1], "range": (0, 255),
                          "label": "YCbCr 色度(Cb) 上限:", "full_idx": 1, "full_list_type": "ycbcr_upper"},
            },
            "ycbcr_Cr": {
                "lower": {"default": self.default_ycbcr_lower_full[2], "range": (0, 255),
                          "label": "YCbCr 饱和度(Cr) 下限:", "full_idx": 2, "full_list_type": "ycbcr_lower"},
                "upper": {"default": self.default_ycbcr_upper_full[2], "range": (0, 255),
                          "label": "YCbCr 饱和度(Cr) 上限:", "full_idx": 2, "full_list_type": "ycbcr_upper"},
            },
        }

        param_order = ["hsv_V", "ycbcr_Y", "ycbcr_Cb", "ycbcr_Cr"]
        grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for i, param_key in enumerate(param_order):
            frame = tk.Frame(self.frame_sliders, padx=5, pady=5)
            frame.grid(row=grid_positions[i][0], column=grid_positions[i][1], sticky="nsew", padx=5, pady=5)

            # 下限
            cfg = self.param_configs[param_key]["lower"]
            tk.Label(frame, text=cfg["label"], font=("Helvetica", 9)).grid(row=0, column=0, sticky="w")
            s = tk.Scale(frame, from_=cfg["range"][0], to=cfg["range"][1], orient=tk.HORIZONTAL, length=120)
            s.set(cfg["default"])
            s.grid(row=0, column=1, padx=2)
            self.scales[f"{param_key}_lower"] = s
            tk.Button(frame, text="-1", width=2, font=("Helvetica", 8),
                      command=lambda s=f"{param_key}_lower": self.scales[s].set(max(0, self.scales[s].get() - 1))).grid(row=0, column=2, padx=1)
            tk.Button(frame, text="+1", width=2, font=("Helvetica", 8),
                      command=lambda s=f"{param_key}_lower": self.scales[s].set(min(255, self.scales[s].get() + 1))).grid(row=0, column=3, padx=1)

            # 上限
            cfg = self.param_configs[param_key]["upper"]
            tk.Label(frame, text=cfg["label"], font=("Helvetica", 9)).grid(row=1, column=0, sticky="w")
            s = tk.Scale(frame, from_=cfg["range"][0], to=cfg["range"][1], orient=tk.HORIZONTAL, length=120)
            s.set(cfg["default"])
            s.grid(row=1, column=1, padx=2)
            self.scales[f"{param_key}_upper"] = s
            tk.Button(frame, text="-1", width=2, font=("Helvetica", 8),
                      command=lambda s=f"{param_key}_upper": self.scales[s].set(max(0, self.scales[s].get() - 1))).grid(row=1, column=2, padx=1)
            tk.Button(frame, text="+1", width=2, font=("Helvetica", 8),
                      command=lambda s=f"{param_key}_upper": self.scales[s].set(min(255, self.scales[s].get() + 1))).grid(row=1, column=3, padx=1)

        self.frame_sliders.columnconfigure(0, weight=1)
        self.frame_sliders.columnconfigure(1, weight=1)
        tk.Button(self.frame_sliders, text="重置参数", command=self.reset_sliders).grid(row=2, column=0, columnspan=2, pady=10)

        # --- 图片显示区 ---
        self.frame_images = tk.Frame(master, padx=10, pady=10)
        self.frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.label_input_image = tk.Label(self.frame_images, text="原始图片")
        self.label_input_image.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.label_output_image = tk.Label(self.frame_images, text="分析结果")
        self.label_output_image.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

    # ---------- 以下逻辑保持不变 ----------
    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="选择人脸图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")])
        if self.image_path:
            self.label_image_path.config(text=self.image_path)
            self.btn_process_image.config(state=tk.NORMAL)
            self.btn_save_image.config(state=tk.DISABLED)
            self.display_image(self.image_path, self.label_input_image)
            self.label_status.config(text="图片已选择，等待处理...")
            self.label_output_image.config(image='', text="分析结果")
            self.original_cv2_image = cv2.imread(self.image_path)
            if self.original_cv2_image is None:
                messagebox.showerror("图片加载错误", "无法加载原始图片进行处理。")
                self.image_path = None
                self.btn_process_image.config(state=tk.DISABLED)

    def display_image(self, path, label_widget, max_size=(500, 500)):
        try:
            img_pil = Image.open(path)
            if label_widget.winfo_width() > 1 and label_widget.winfo_height() > 1:
                max_size = (label_widget.winfo_width(), label_widget.winfo_height())
            img_pil.thumbnail(max_size, Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_widget.config(image=img_tk, text="")
            label_widget.image = img_tk
        except Exception as e:
            messagebox.showerror("图片显示错误", f"无法显示图片: {e}")
            label_widget.config(image='', text="图片加载失败")

    def display_cv2_image(self, cv2_img, label_widget, max_size=(500, 500)):
        try:
            cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(cv2_img_rgb)
            if label_widget.winfo_width() > 1 and label_widget.winfo_height() > 1:
                max_size = (label_widget.winfo_width(), label_widget.winfo_height())
            img_pil.thumbnail(max_size, Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img_pil)
            label_widget.config(image=img_tk, text="")
            label_widget.image = img_tk
            self.processed_cv2_image = cv2_img
            self.btn_save_image.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("图片显示错误", f"无法显示处理结果: {e}")
            label_widget.config(image='', text="结果显示失败")
            self.btn_save_image.config(state=tk.DISABLED)

    def get_slider_params(self):
        hsv_lower = list(self.default_hsv_lower_full)
        hsv_lower[2] = self.scales["hsv_V_lower"].get()

        hsv_upper = list(self.default_hsv_upper_full)
        hsv_upper[2] = self.scales["hsv_V_upper"].get()

        ycbcr_lower = list(self.default_ycbcr_lower_full)
        ycbcr_lower[0] = self.scales["ycbcr_Y_lower"].get()
        ycbcr_lower[1] = self.scales["ycbcr_Cb_lower"].get()
        ycbcr_lower[2] = self.scales["ycbcr_Cr_lower"].get()

        ycbcr_upper = list(self.default_ycbcr_upper_full)
        ycbcr_upper[0] = self.scales["ycbcr_Y_upper"].get()
        ycbcr_upper[1] = self.scales["ycbcr_Cb_upper"].get()
        ycbcr_upper[2] = self.scales["ycbcr_Cr_upper"].get()

        return hsv_lower, hsv_upper, ycbcr_lower, ycbcr_upper

    def reset_sliders(self):
        for param_key in self.param_configs:
            for bound in ("lower", "upper"):
                slider_name = f"{param_key}_{bound}"
                default_val = self.param_configs[param_key][bound]["default"]
                self.scales[slider_name].set(default_val)

    def process_image(self):
        if not self.image_path:
            messagebox.showwarning("提示", "请先选择一张图片！")
            return

        self.label_status.config(text="正在处理图片，请稍候...", fg="orange")
        self.master.update_idletasks()

        hsv_lower, hsv_upper, ycbcr_lower, ycbcr_upper = self.get_slider_params()
        try:
            processed_img = process_and_draw_face_geometry(
                self.image_path,
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                ycbcr_lower=ycbcr_lower,
                ycbcr_upper=ycbcr_upper)
            self.display_cv2_image(processed_img, self.label_output_image)
            self.label_status.config(text="处理完成！", fg="green")
        except Exception as e:
            messagebox.showerror("处理错误", str(e))
            self.label_status.config(text="处理失败", fg="red")

    def save_image(self):
        if self.processed_cv2_image is None:
            messagebox.showwarning("提示", "没有可保存的处理结果图片。")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="保存处理结果图片")
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_cv2_image)
                messagebox.showinfo("保存成功", f"图片已保存到: {file_path}")
            except Exception as e:
                messagebox.showerror("保存失败", f"保存图片时发生错误: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceGeometryApp(root)
    root.mainloop()