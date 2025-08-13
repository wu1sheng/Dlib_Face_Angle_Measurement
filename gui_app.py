import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

import importlib

# 您封装的处理函数
from face_processor  import process_and_draw_face_geometry


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
        master.geometry("1920x1200")             # 初始高度减小
        master.resizable(True, True)

        self.image_path = None
        self.original_cv2_image = None
        self.processed_cv2_image = None
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_center = None
        self._current_roi = None
        self.input_image_tk = None
        self.output_image_tk = None

        # 默认肤色检测参数（完整）
        self.default_hsv_lower_full = [5, 53, 92]
        self.default_hsv_upper_full = [18, 140, 225]
        self.default_ycbcr_lower_full = [95, 135, 80]
        self.default_ycbcr_upper_full = [220, 165, 116]

        # 参考真实宽度（mm），默认 170mm（17cm）
        self.ref_width_var = tk.StringVar(value="170")


        # --- 顶部控制区 ---
        self.frame_controls = tk.Frame(master, padx=10, pady=10)
        self.frame_controls.pack(side=tk.TOP, fill=tk.X)

        tk.Label(self.frame_controls, text="人脸几何特征分析", font=("Helvetica", 16, "bold")).pack(pady=5)

        self.btn_select_image = tk.Button(self.frame_controls, text="选择图片", command=self.select_image)
        self.btn_select_image.pack(side=tk.LEFT, padx=5)

        # 在 self.btn_save_image.pack(...) 之后，插入 ↓
        self.angle_frame = tk.LabelFrame(self.frame_controls, text="下颌角度", padx=5, pady=5)
        self.angle_frame.pack(side=tk.RIGHT, padx=10)
        self.angle_text = tk.Text(self.angle_frame, width=20, height=3,
                                  state="disabled", font=("Consolas", 10))
        self.angle_text.pack()

        self.label_image_path = tk.Label(self.frame_controls, text="未选择图片", width=60, anchor="w")
        self.label_image_path.pack(side=tk.LEFT, padx=5)

        self.btn_process_image = tk.Button(self.frame_controls, text="处理图片", command=self.process_image, state=tk.DISABLED)
        self.btn_process_image.pack(side=tk.LEFT, padx=5)

        self.btn_save_image = tk.Button(self.frame_controls, text="保存图片", command=self.save_image, state=tk.DISABLED)
        self.btn_save_image.pack(side=tk.LEFT, padx=5)

        # 参考真实宽度输入
        tk.Label(self.frame_controls, text="参考真实宽度(mm):").pack(side=tk.LEFT, padx=4)
        tk.Entry(self.frame_controls, textvariable=self.ref_width_var, width=8).pack(side=tk.LEFT)

        # 其它几何量显示
        self.metrics_frame = tk.LabelFrame(self.frame_controls, text="几何量", padx=5, pady=5)
        self.metrics_frame.pack(side=tk.RIGHT, padx=10)
        self.metrics_text = tk.Text(self.metrics_frame, width=36, height=4, state="disabled", font=("Consolas", 10))
        self.metrics_text.pack()


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
                          "label": "YCbCr 亮度(Y) 下限（脸黑调小，脸白调高）:", "full_idx": 0, "full_list_type": "ycbcr_lower"},
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

        # 局部缩放（仅放大局部，显示大小不变）
        self.label_output_image.bind("<Button-1>", self.on_output_click)
        self.label_output_image.bind("<Button-3>", self.on_output_right_press)
        self.label_output_image.bind("<B3-Motion>", self.on_output_right_drag)
        self.label_output_image.bind("<ButtonRelease-3>", self.on_output_right_release)
        self.zoom_bar = tk.Scale(self.frame_images, from_=1.0, to=4.0, resolution=0.1,
                                 orient=tk.HORIZONTAL, label="局部缩放",
                                 variable=self.zoom_var, command=lambda v: self.redraw_output_with_zoom())
        self.zoom_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10)

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

    # ---- 新增：根据 label 剩余空间计算等比例尺寸 ----
    def _fit_size(self, img_pil, label_widget):
        # 标签实际可用像素（留 20 px 边距）
        avail_w = max(label_widget.winfo_width()  - 20, 1)
        avail_h = max(label_widget.winfo_height() - 20, 1)
        img_w, img_h = img_pil.size
        scale = min(avail_w / img_w, avail_h / img_h, 1)   # 不放大
        return int(img_w * scale), int(img_h * scale)

    def display_image(self, path, label_widget):
        try:
            img_pil = Image.open(path)
            img_pil = img_pil.resize(self._fit_size(img_pil, label_widget), Image.LANCZOS)
            img_tk  = ImageTk.PhotoImage(img_pil)
            label_widget.config(image=img_tk, text="")
            label_widget.image = img_tk
        except Exception as e:
            messagebox.showerror("图片显示错误", f"无法显示图片: {e}")
            label_widget.config(image='', text="图片加载失败")

    
    def _fit_size(self, img_pil, label_widget):
        lw = max(1, label_widget.winfo_width())
        lh = max(1, label_widget.winfo_height())
        iw, ih = img_pil.size
        scale = min(lw/iw, lh/ih)
        return (max(1, int(iw*scale)), max(1, int(ih*scale)))

    def _render_output(self, cv2_img, label_widget):
        from PIL import Image
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        ih, iw = cv2_img_rgb.shape[:2]
        full_pil = Image.fromarray(cv2_img_rgb)
        # final display size
        fit_w, fit_h = self._fit_size(full_pil, label_widget)
        z = max(1.0, float(self.zoom_var.get()))
        crop_w = max(1, int(iw / z)); crop_h = max(1, int(ih / z))
        # center
        if self.zoom_center is None:
            cx, cy = iw//2, ih//2
        else:
            cx, cy = self.zoom_center
        x1 = max(0, min(iw - crop_w, int(cx - crop_w//2)))
        y1 = max(0, min(ih - crop_h, int(cy - crop_h//2)))
        roi = full_pil.crop((x1, y1, x1+crop_w, y1+crop_h))
        roi = roi.resize((fit_w, fit_h), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(roi)
        label_widget.config(image=tkimg, text="")
        label_widget.image = tkimg
        self._current_roi = (x1, y1, crop_w, crop_h, fit_w, fit_h, iw, ih)

    def _render_output(self, cv2_img, label_widget):
        from PIL import Image
        cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        ih, iw = cv2_img_rgb.shape[:2]
        full_pil = Image.fromarray(cv2_img_rgb)
        fit_w, fit_h = self._fit_size(full_pil, label_widget)
        z = max(1.0, float(self.zoom_var.get()))
        crop_w = max(1, int(iw / z)); crop_h = max(1, int(ih / z))
        if self.zoom_center is None:
            cx, cy = iw//2, ih//2
        else:
            cx, cy = self.zoom_center
        x1 = max(0, min(iw - crop_w, int(cx - crop_w//2)))
        y1 = max(0, min(ih - crop_h, int(cy - crop_h//2)))
        roi = full_pil.crop((x1, y1, x1+crop_w, y1+crop_h))
        roi = roi.resize((fit_w, fit_h), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(roi)
        label_widget.config(image=tkimg, text="")
        label_widget.image = tkimg
        self._current_roi = (x1, y1, crop_w, crop_h, fit_w, fit_h, iw, ih)

    def display_cv2_image(self, cv2_img, label_widget):
        try:
            self._render_output(cv2_img, label_widget)
            self.processed_cv2_image = cv2_img
            self.btn_save_image.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("图片显示错误", f"无法显示处理结果: {e}")
            label_widget.config(image='', text="结果显示失败")
            self.btn_save_image.config(state=tk.DISABLED)

    def redraw_output_with_zoom(self):
        if self.processed_cv2_image is not None:
            try:
                self._render_output(self.processed_cv2_image, self.label_output_image)
            except Exception:
                pass

    def on_output_click(self, event):
        if self.processed_cv2_image is None or not self._current_roi:
            return
        x1, y1, crop_w, crop_h, fit_w, fit_h, iw, ih = self._current_roi
        u = max(0, min(fit_w-1, event.x)); v = max(0, min(fit_h-1, event.y))
        rx = int(x1 + (u/ max(1, fit_w)) * crop_w)
        ry = int(y1 + (v/ max(1, fit_h)) * crop_h)
        self.zoom_center = (rx, ry)
        self.redraw_output_with_zoom()

    def on_output_right_press(self, event):
        self._dragging = True
        self._drag_last = (event.x, event.y)

    def on_output_right_drag(self, event):
        if not getattr(self, "_dragging", False) or not self._current_roi:
            return
        x1, y1, crop_w, crop_h, fit_w, fit_h, iw, ih = self._current_roi
        last_x, last_y = getattr(self, "_drag_last", (event.x, event.y))
        dx = event.x - last_x; dy = event.y - last_y
        if fit_w < 1 or fit_h < 1:
            return
        shift_x = int(dx / max(1, fit_w) * crop_w)
        shift_y = int(dy / max(1, fit_h) * crop_h)
        if self.zoom_center is None:
            cx, cy = iw//2, ih//2
        else:
            cx, cy = self.zoom_center
        cx = max(0, min(iw-1, cx - shift_x))
        cy = max(0, min(ih-1, cy - shift_y))
        self.zoom_center = (cx, cy)
        self._drag_last = (event.x, event.y)
        self.redraw_output_with_zoom()

    def on_output_right_release(self, event):
        self._dragging = False
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
            processed_img, left_angle, right_angle, metrics = process_and_draw_face_geometry(
                self.image_path,
                hsv_lower=hsv_lower,
                hsv_upper=hsv_upper,
                ycbcr_lower=ycbcr_lower,
                ycbcr_upper=ycbcr_upper,
                ref_real_width_mm_at_row=(float(self.ref_width_var.get()) if self.ref_width_var.get().strip() else 170.0))
            self.display_cv2_image(processed_img, self.label_output_image)
            # reset zoom center
            if processed_img is not None:
                ih, iw = processed_img.shape[:2]
                self.zoom_center = (iw//2, ih//2)
                self.zoom_var.set(1.0)
                self.redraw_output_with_zoom()
            self.label_status.config(text="处理完成！", fg="green")
            self.angle_text.configure(state="normal")
            self.angle_text.delete(1.0, tk.END)
            self.angle_text.insert(tk.END,
                                   f"左角: {left_angle:.3f}°\n右角: {right_angle:.3f}°\n夹角: {right_angle+left_angle:.3f}°")
            self.angle_text.configure(state="disabled")
            # 显示几何量
            self.metrics_text.configure(state="normal"); self.metrics_text.delete(1.0, tk.END)
            lw_px = metrics.get('lower_face_width_px')
            lw_mm = metrics.get('lower_face_width_mm')
            side = metrics.get('side_angles_deg', {})
            def fmt(v):
                try:
                    return f"{float(v):.3f}"
                except:
                    return "--"
            lw_cm = (lw_mm/10.0) if lw_mm is not None else None
            self.metrics_text.insert(tk.END, f"下半面宽度: {fmt(lw_cm)} cm\n")
            self.metrics_text.insert(tk.END, f"左侧角度(补角): {fmt(side.get('left'))}°\n")
            self.metrics_text.insert(tk.END, f"右侧角度(补角): {fmt(side.get('right'))}°\n")
            self.metrics_text.configure(state="disabled")

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