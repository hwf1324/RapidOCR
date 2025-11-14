# -*- encoding: utf-8 -*-
# @Author: SWHL, hwf1324
# @Contact: liekkaskono@163.com, 1398969445@qq.com

import sys
import os
import shutil
from ctypes import Array
from typing import TypeVar, override

sys.path.insert(0, os.path.dirname(__file__) + "\\lib")

import globalPluginHandler
import gui.guiHelper
import numpy as np
import ui
import wx
import winUser
from contentRecog import ContentRecognizer, LinesWordsResult, RecogImageInfo, onRecognizeResultCallbackT
from locationHelper import Point, RectLTWH
from scriptHandler import script
from winBindings.gdi32 import RGBQUAD

from .main import RapidOCR, DEFAULT_CFG_PATH, USER_CFG_PATH
from .utils.log import logger
from .utils.output import RapidOCROutput
from .utils.parse_parameters import ParseParams
from .utils.typings import EngineType, LangCls, LangDet, LangRec, ModelType, OCRVersion


gui.guiHelper._HorizontalCtrlT = TypeVar(
	"_HorizontalCtrlT",
	wx.Button,
	wx.Choice,
	wx.ComboBox,
	wx.Slider,
	wx.SpinCtrl,
	wx.TextCtrl,
	wx.SpinCtrlDouble,
)


engine: RapidOCR | None = None
try:
	if not os.path.exists(USER_CFG_PATH):
		shutil.copyfile(DEFAULT_CFG_PATH, USER_CFG_PATH)
	cfg = ParseParams.load(USER_CFG_PATH)
except:
	cfg = ParseParams.load(DEFAULT_CFG_PATH)


def rgbquad_array_to_bgr_ndarray(pixels: Array[RGBQUAD], width: int, height: int) -> np.ndarray:
	"""
	将 ctypes RGBQUAD 数组转换为BGR格式的numpy数组

	Parameters:
	pixels: ctypes.Array[RGBQUAD] - RGBQUAD结构体数组
	width: int - 图像宽度
	height: int - 图像高度

	Returns:
	np.ndarray - BGR格式的图像数组，形状为(height, width, 3)
	"""
	# 将 ctypes 数组转换为 numpy 数组
	# RGBQUAD 结构体通常包含 4 个字节：Blue, Green, Red, Reserved
	# 注意：Windows 中的 RGBQUAD 实际上是 BGR 顺序存储的
	rgb_array = np.ctypeslib.as_array(pixels, shape=(height, width))

	# 创建一个新的数组来存储 BGR 数据
	# 由于 RGBQUAD 本身已经是 BGR 顺序，我们只需要提取 RGB 通道
	bgr_image = np.zeros((height, width, 3), dtype=np.uint8)

	# 提取各通道数据
	bgr_image[:, :, 0] = rgb_array["rgbBlue"]  # B 通道
	bgr_image[:, :, 1] = rgb_array["rgbGreen"]  # G 通道
	bgr_image[:, :, 2] = rgb_array["rgbRed"]  # R 通道

	return bgr_image


class PaddleOCR(ContentRecognizer):

	@override
	def recognize(self, pixels: Array[RGBQUAD], imageInfo: RecogImageInfo, onResult: onRecognizeResultCallbackT):
		self._onResult = onResult
		width = imageInfo.recogWidth
		height = imageInfo.recogHeight
		image = rgbquad_array_to_bgr_ndarray(pixels, width, height)
		# image = Image.frombuffer("RGBX", (width, height), pixels, "raw", "BGRX", 0, 1)

		global engine
		if engine is None:
			engine = RapidOCR()
		result = engine(image)
		if isinstance(result, RapidOCROutput) and result.txts:
			# ui.message(
			# 	f"整体耗时 {result.elapse:.2f} 秒、"
			# 	+ f"检测耗时 {result.elapse_list[0]:.2f} 秒、"
			# 	+ f"分类耗时 {result.elapse_list[1]:.2f} 秒、"
			# 	+ f"识别耗时 {result.elapse_list[2]:.2f} 秒"
			# )
			# word_results = result.word_results[0]
			lines: list[list[dict]] = []
			if engine.return_word_box:
				for line in result.word_results:
					words: list[dict] = []
					for char in line:
						rect: RectLTWH = RectLTWH.fromCollection(*[Point(point[0], point[1]) for point in char[2]])
						words.append({"x": rect.left, "y": rect.top, "width": rect.width, "height": rect.height, "text": char[0]})
			else:
				for box, text in zip(result.boxes.tolist(), result.txts):
					rect: RectLTWH = RectLTWH.fromCollection(*[Point.fromFloatCollection(*point) for point in box])
					lines.append([{"x": rect.left, "y": rect.top, "width": rect.width, "height": rect.height, "text": text}])

			self._onResult(LinesWordsResult(lines, imageInfo))
		else:
			self._onResult(RuntimeError("Paddle OCR failed"))

	@override
	def cancel(self) -> None:
		self._onResult = None


def GetDescription(self: wx.Accessible, childId: int):
	if childId == winUser.CHILDID_SELF:
		if hasattr(self.Window, "description"):
			return (wx.ACC_OK, self.Window.description)
		else:
			return super().GetDescription(childId)


def setDescription(window: wx.Window, description: str) -> None:
	window.GetAccessible().GetDescription = GetDescription
	window.description = description


class PaddleOCRSettingsPanel(gui.settingsDialogs.SettingsPanel):
	"""Panel for Paddle OCR settings."""

	title = _("Paddle OCR")

	@override
	def makeSettings(self, sizer: wx.BoxSizer):
		sHelper = gui.guiHelper.BoxSizerHelper(self, sizer=sizer)
		generalGroupSizer = wx.StaticBoxSizer(wx.VERTICAL, self, label=_("General"))
		generalBox = generalGroupSizer.GetStaticBox()
		generalGroup = gui.guiHelper.BoxSizerHelper(self, sizer=generalGroupSizer)
		sHelper.addItem(generalGroup)

		self.useDet = generalGroup.addItem(wx.CheckBox(generalBox, label=_("Use detection:")))
		self.useDet.SetValue(cfg.Global.use_det)

		self.useCls = generalGroup.addItem(wx.CheckBox(generalBox, label=_("Use classification:")))
		self.useCls.SetValue(cfg.Global.use_cls)

		self.useRec = generalGroup.addItem(wx.CheckBox(generalBox, label=_("Use recognition:")))
		self.useRec.SetValue(cfg.Global.use_rec)

		self.textScore = generalGroup.addLabeledControl(
			_("Text score:"),
			wx.SpinCtrlDouble,
			max=1.0,
			initial=cfg.Global.text_score,
			inc=0.1
		)

		self.wordBox = generalGroup.addItem(wx.CheckBox(generalBox, label=_("Return word boxes:")))
		self.wordBox.SetValue(cfg.Global.return_word_box)

		self.singleCharBox = generalGroup.addItem(wx.CheckBox(generalBox, label=_("Return single character boxes:")))
		self.singleCharBox.SetValue(cfg.Global.return_single_char_box)

		self.minHeight = generalGroup.addLabeledControl(
			_("Minimum height for direct image recognition (pixels):"),
			wx.SpinCtrl,
			max=1000,
			initial=cfg.Global.min_height,
		)

		self.widthHeightRatio = generalGroup.addLabeledControl(
			_("Minimum width/height ratio for direct image recognition:"),
			wx.SpinCtrlDouble,
			initial=cfg.Global.width_height_ratio,
		)

		self.maxSideLen = generalGroup.addLabeledControl(
			_("Maximum image side length (scaled proportionally if exceeded) (pixels):"),
			wx.SpinCtrl,
			max=10000,
			initial=cfg.Global.max_side_len,
		)

		self.minSideLen = generalGroup.addLabeledControl(
			_("Minimum image side length (scaled proportionally if smaller) (pixels):"),
			wx.SpinCtrl,
			max=10000,
			initial=cfg.Global.min_side_len,
		)

		detGroupSizer = wx.StaticBoxSizer(wx.VERTICAL, self, label=_("Detection"))
		detBox = detGroupSizer.GetStaticBox()
		self.detGroup = gui.guiHelper.BoxSizerHelper(self, sizer=detGroupSizer)
		sHelper.addItem(self.detGroup)

		self.detLan = self.detGroup.addLabeledControl(
			_("Language:"),
			wx.Choice,
			choices=[lang.value for lang in LangDet],
		)
		self.detLan.SetSelection(list(LangDet).index(cfg.Det.lang_type))

		self.detModel = self.detGroup.addLabeledControl(
			_("Model:"),
			wx.Choice,
			choices=[model.value for model in ModelType],
		)
		self.detModel.SetSelection(list(ModelType).index(cfg.Det.model_type))

		self.detVersion = self.detGroup.addLabeledControl(
			_("Version:"),
			wx.Choice,
			choices=[version.value for version in OCRVersion],
		)
		self.detVersion.SetSelection(list(OCRVersion).index(cfg.Det.ocr_version))

		clsGroupSizer = wx.StaticBoxSizer(wx.VERTICAL, self, label=_("Classification"))
		clsBox = clsGroupSizer.GetStaticBox()
		self.clsGroup = gui.guiHelper.BoxSizerHelper(self, sizer=clsGroupSizer)
		sHelper.addItem(self.clsGroup)

		self.clsLan = self.clsGroup.addLabeledControl(
			_("Language:"),
			wx.Choice,
			choices=[lang.value for lang in LangCls],
		)
		self.clsLan.SetSelection(list(LangCls).index(cfg.Cls.lang_type))

		self.clsModel = self.clsGroup.addLabeledControl(
			_("Model:"),
			wx.Choice,
			choices=[model.value for model in ModelType],
		)
		self.clsModel.SetSelection(list(ModelType).index(cfg.Cls.model_type))

		self.clsVersion = self.clsGroup.addLabeledControl(
			_("Version:"),
			wx.Choice,
			choices=[version.value for version in OCRVersion],
		)
		self.clsVersion.SetSelection(list(OCRVersion).index(cfg.Cls.ocr_version))

		recGroupSizer = wx.StaticBoxSizer(wx.VERTICAL, self, label=_("Recognition"))
		recBox = recGroupSizer.GetStaticBox()
		self.recGroup = gui.guiHelper.BoxSizerHelper(self, sizer=recGroupSizer)
		sHelper.addItem(self.recGroup)

		self.recLan = self.recGroup.addLabeledControl(
			_("Language:"),
			wx.Choice,
			choices=[lang.value for lang in LangRec],
		)
		self.recLan.SetSelection(list(LangRec).index(cfg.Rec.lang_type))

		self.recModel = self.recGroup.addLabeledControl(
			_("Model:"),
			wx.Choice,
			choices=[model.value for model in ModelType],
		)
		self.recModel.SetSelection(list(ModelType).index(cfg.Rec.model_type))

		self.recVersion = self.recGroup.addLabeledControl(
			_("Version:"),
			wx.Choice,
			choices=[version.value for version in OCRVersion],
		)
		self.recVersion.SetSelection(list(OCRVersion).index(cfg.Rec.ocr_version))

		# self.Bind(wx.EVT_CHECKBOX, self.onCheckCtrl)

	@override
	def onSave(self):
		cfg.Global.use_det = self.useDet.GetValue()
		cfg.Global.use_cls = self.useCls.GetValue()
		cfg.Global.use_rec = self.useRec.GetValue()
		cfg.Global.text_score = self.textScore.GetValue()
		cfg.Global.return_word_box = self.wordBox.GetValue()
		cfg.Global.return_single_char_box = self.singleCharBox.GetValue()
		cfg.Global.min_height = self.minHeight.GetValue()
		cfg.Global.width_height_ratio = self.widthHeightRatio.GetValue()
		cfg.Global.max_side_len = self.maxSideLen.GetValue()
		cfg.Global.min_side_len = self.minSideLen.GetValue()

		cfg.Det.lang_type = list(LangDet)[self.detLan.GetSelection()]
		cfg.Det.model_type = list(ModelType)[self.detModel.GetSelection()]
		cfg.Det.ocr_version = list(OCRVersion)[self.detVersion.GetSelection()]

		cfg.Cls.lang_type = list(LangCls)[self.clsLan.GetSelection()]
		cfg.Cls.model_type = list(ModelType)[self.clsModel.GetSelection()]
		cfg.Cls.ocr_version = list(OCRVersion)[self.clsVersion.GetSelection()]

		cfg.Rec.lang_type = list(LangRec)[self.recLan.GetSelection()]
		cfg.Rec.model_type = list(ModelType)[self.recModel.GetSelection()]
		cfg.Rec.ocr_version = list(OCRVersion)[self.recVersion.GetSelection()]

		ParseParams.save(cfg, USER_CFG_PATH)
		global engine
		engine = None

	# def onCheckCtrl(self):
	# 	self.singleCharBox.Enable(self.wordBox.GetValue())
	# 	self.detGroup.enable(self.useDet.GetValue())
	# 	# self.clsGroup.enable(self.useCls.GetValue())
	# 	self.recGroup.enable(self.useRec.GetValue())


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	def __init__(self):
		super().__init__()
		gui.settingsDialogs.NVDASettingsDialog.categoryClasses.append(PaddleOCRSettingsPanel)

	@override
	def terminate(self):
		gui.settingsDialogs.NVDASettingsDialog.categoryClasses.remove(PaddleOCRSettingsPanel)
		global engine
		engine = None
		return super().terminate()

	@script(
		# Translators: Describes a command.
		description=_("Recognizes the content of the current navigator object with Paddle OCR"),
		gesture="kb:NVDA+r",
	)
	def script_recognizeWithPaddleOcr(self, gesture):
		import vision
		from visionEnhancementProviders.screenCurtain import ScreenCurtainProvider

		screenCurtainId = ScreenCurtainProvider.getSettings().getId()
		screenCurtainProviderInfo = vision.handler.getProviderInfo(screenCurtainId)
		isScreenCurtainRunning = bool(vision.handler.getProviderInstance(screenCurtainProviderInfo))
		if isScreenCurtainRunning:
			# Translators: Reported when screen curtain is enabled.
			ui.message(_("Please disable screen curtain before using Paddle OCR."))
			return
		from contentRecog import recogUi

		global engine
		if engine is None:
			engine = RapidOCR()
		recog = PaddleOCR(engine)
		recogUi.recognizeNavigatorObject(recog)
