from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable, KeepTogether,
)
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pdf' / 'sgg_survey_notes.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('CNBold', 'C:/Windows/Fonts/msyhbd.ttc'))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNBold', italic='CN', boldItalic='CNBold')

INK = colors.HexColor('#18232B')
GRAY = colors.HexColor('#5C6670')
BLUE = colors.HexColor('#DCE9F1')
EDGE = colors.HexColor('#D9D9D9')
WIDTH = letter[0] - 104
styles = {
    'body': ParagraphStyle('Body', fontName='CN', fontSize=11, leading=17,
                           textColor=INK, spaceAfter=7, wordWrap='CJK'),
    'small': ParagraphStyle('Small', fontName='CN', fontSize=9.2, leading=14,
                            textColor=GRAY, spaceAfter=5, wordWrap='CJK'),
    'title': ParagraphStyle('Title', fontName='CNBold', fontSize=21, leading=30,
                            textColor=colors.black, spaceAfter=8),
    'subtitle': ParagraphStyle('Subtitle', fontName='CN', fontSize=11, leading=17,
                               textColor=colors.black, spaceAfter=14),
    'h1': ParagraphStyle('H1', fontName='CNBold', fontSize=14, leading=22,
                         textColor=colors.black, spaceBefore=8, spaceAfter=8,
                         keepWithNext=True),
    'h2': ParagraphStyle('H2', fontName='CNBold', fontSize=11.3, leading=18,
                         textColor=colors.black, spaceBefore=7, spaceAfter=5,
                         keepWithNext=True),
    'cell': ParagraphStyle('Cell', fontName='CN', fontSize=10.1, leading=15.2,
                           textColor=INK, wordWrap='CJK'),
    'head': ParagraphStyle('Head', fontName='CNBold', fontSize=10.2, leading=15,
                           textColor=colors.black, wordWrap='CJK'),
    'math': ParagraphStyle('Math', fontName='CN', fontSize=11.5, leading=23,
                           alignment=TA_CENTER, spaceBefore=7, spaceAfter=10),
    'quiz': ParagraphStyle('Quiz', fontName='CN', fontSize=10.4, leading=16.1,
                           textColor=INK, spaceAfter=6, wordWrap='CJK'),
}
story = []


def p(text, style='body'):
    story.append(Paragraph(text, styles[style]))


def h(text, level=1):
    p(text, 'h1' if level == 1 else 'h2')


def table(headers, rows, widths):
    data = [[Paragraph(escape(str(v)), styles['head']) for v in headers]]
    data += [[Paragraph(escape(str(v)), styles['cell']) for v in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, EDGE),
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 9))


def newpage():
    story.append(PageBreak())


class SceneDiagram(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 86

    def draw(self):
        c = self.canv
        y = 27
        xs = [6, 204, 402]
        for x, label in zip(xs, ['人', '马', '草地']):
            c.setStrokeColor(colors.HexColor('#8EA7B7'))
            c.setFillColor(colors.HexColor('#F3F7FA'))
            c.roundRect(x, y, 94, 36, 8, fill=1, stroke=1)
            c.setFont('CNBold', 12)
            c.setFillColor(INK)
            c.drawCentredString(x + 47, y + 12, label)
        for x1, x2, label in [(100, 204, '骑'), (298, 402, '站在')]:
            c.setStrokeColor(INK)
            c.setFillColor(INK)
            c.setLineWidth(1)
            c.line(x1 + 4, y + 18, x2 - 5, y + 18)
            path = c.beginPath()
            path.moveTo(x2 - 5, y + 18)
            path.lineTo(x2 - 11, y + 21)
            path.lineTo(x2 - 11, y + 15)
            path.close()
            c.drawPath(path, fill=1, stroke=0)
            c.setFont('CN', 10)
            c.drawCentredString((x1 + x2) / 2, y + 28, label)
        c.setFont('CN', 9)
        c.setFillColor(GRAY)
        c.drawCentredString(WIDTH / 2, 6, '节点表示对象实例  箭头表示有方向的关系')


class RecallFraction(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 56

    def draw(self):
        c = self.canv
        c.setFillColor(INK)
        c.setStrokeColor(INK)
        c.setFont('CN', 11)
        c.drawString(30, 24, 'R@K =')
        x, w = 108, 345
        c.drawCentredString(x + w / 2, 37, '前 K 条预测中匹配到的真实关系数')
        c.line(x, 31, x + w, 31)
        c.drawCentredString(x + w / 2, 11, '真实关系总数')


# Page 1
p('SGG综述中文阅读笔记', 'title')
p('从基础概念到方法 评测与研究难点', 'subtitle')
p('场景图生成把图片中的对象及其关系组织成结构化表示。本笔记面向刚接触深度学习的读者，以“人骑马，马站在草地上”为共同例子，建立阅读综述和理解后续论文所需的基本框架。')
p('对应论文：Guangming Zhu 等，Scene Graph Generation: A Comprehensive Survey，arXiv:2201.00443v2，2022 年 6 月 22 日。该版本综述 138 篇代表性工作，主要从特征表示与特征优化组织方法。', 'small')
h('1 基本概念')
p('<b>目标检测</b>（Object Detection）回答对象在哪里、是什么。<b>场景图生成</b>（Scene Graph Generation，SGG）进一步预测对象之间如何联系。SGG 的输入是图像，输出是预测得到的对象及关系结构；人工标注为训练和评价提供参考答案。')
table(['概念', '含义与例子'], [
    ['检测框 Bounding Box', '用矩形区域定位对象，例如框住图片中的人或马。'],
    ['对象类别 Object Class', '说明区域中的对象是什么，例如“人”“马”。'],
    ['特征向量 Feature Vector', '模型从图像区域提取的一串数字，用于描述有助于识别和关系判断的信息。各维通常没有固定的人工语义。'],
    ['关系三元组 Relationship Triplet', '以〈主体，关系，客体〉表达联系，例如〈人，骑，马〉。'],
], [158, WIDTH - 158])
story.append(SceneDiagram())
p('场景图的节点对应具体对象实例。例如两个人应有两个节点，即使它们都属于“人”这一类别。箭头方向也有含义：〈人，骑，马〉与〈马，骑，人〉不同。')
p('读图要点：检测框提供单个对象的位置；空间关系描述两个对象如何排列；动作关系还需要结合姿态等视觉证据。“人在马上方”是判断“骑”的线索，但不能单凭这一点确定关系。')
p('原文定位：引言与第 2 节。图示和中文关系词为教学例子；实际类别与关系词表以具体数据集为准。', 'small')

# Page 2
newpage()
h('2 生成流程与数学符号')
p('先把任务拆成三个相互依赖的部分：找到对象位置，判断对象类别，再预测对象间关系。收集预测的关系三元组后，可以组成场景图。')
p('<b>图片 → 检测框 B → 对象类别 O → 关系 R → 场景图 T</b>', 'math')
p('例如，模型先定位人和马，再识别两个区域的类别，最后结合区域特征、姿态、相对位置与上下文，判断两者是“骑”“牵着”还是其他关系。前两部分合起来对应目标检测。')
h('符号怎样记', 2)
table(['符号', '本文中的含义', '辅助记忆'], [
    ['S', '输入的视觉场景，这里可理解为图片', 'Scene'],
    ['B', '对象候选检测框集合', 'Bounding boxes'],
    ['O', '公式中的对象类别信息', 'Objects'],
    ['R', '对象之间的关系集合', 'Relations'],
    ['T', '场景图', '按作者定义理解，不强行展开英文词'],
    ['p', '概率', 'Probability'],
    ['竖线 |', '在已知右侧信息的条件下', '条件概率记号'],
], [48, 272, WIDTH - 320])
h('公式 2 的直观含义', 2)
p('略去原文下标，可将公式写成：')
p('p(T | S) = p(B | S) p(O | B, S) p(R | O, B, S)', 'math')
p('<b>p(B | S)</b>：看着图片，预测对象的位置。<br/><b>p(O | B, S)</b>：已知图片和位置，预测对象类别。<br/><b>p(R | O, B, S)</b>：已知图片、位置和类别，预测关系。')
p('这是概率分解，用来表达各预测部分的条件依赖。三部分并非相互独立；具体模型也可以联合预测这些信息，而不必实现成三个独立运行的程序。')
p('常见错误：如果把马识别为自行车，即使“骑”这个谓词没有变化，完整预测〈人，骑，自行车〉仍然是错的。对象类别错误可能影响后续关系判断，也会影响整个三元组是否正确。')
p('原文定位：第 3 节开头，公式（2）。英文字母是作者定义的记号，跨论文阅读时应重新确认其含义。', 'small')

# Page 3
newpage()
h('3 模型使用哪些信息')
p('上一节说明要预测什么；这一节关注模型凭哪些证据作判断。不同信息可以组合使用。')
table(['信息', '在同一场景中的作用'], [
    ['外观 Appearance', '观察人的姿态、马的形状以及人与马接触区域的视觉内容。'],
    ['空间 Spatial', '利用两个检测框的相对位置、大小和排列。'],
    ['语义 Semantic', '利用对象类别和关系名称的含义。'],
    ['上下文 Context', '结合周围对象和整体场景，例如马鞍等额外线索。'],
], [132, WIDTH - 132])
p('<b>先验信息</b>（Prior Information）包含统计先验和语言先验。前者来自训练数据中的出现频率；后者利用词语或类别之间的语义联系。<b>常识知识</b>（Commonsense Knowledge）提供对象属性、事件和交互的一般知识，可以来自训练数据，也可以来自外部知识库。')
p('经验能够帮助判断，但“经常发生”并不等于“当前画面正在发生”。假设训练中“人骑马”很常见，模型也仍需根据当前图像区分骑马和牵马。')
h('4 三种关系建模机制')
h('消息传递 Message Passing', 2)
p('模型汇总其他对象或关系的信息，用于更新当前表示。例如，更新“人”的特征时，结合“马”和周围场景的特征。消息可以在三元组内部传递，也可以覆盖更广的上下文；实际传递的是数字特征。')
h('注意力 Attention', 2)
p('模型为不同区域、特征或上下文分配不同权重。判断人是否骑马时，人与马的交互区域可能比远处草地更有帮助。注意力也可以用于消息传递，决定各部分信息的影响程度。')
h('视觉平移嵌入 Visual Translation Embedding', 2)
p('嵌入把对象等信息映射为学习空间中的向量。VTransE 的直观约束是“主体表示加上关系向量，接近客体表示”。原文公式（9）写为：')
p('W<sub>s</sub>x<sub>s</sub> + t<sub>p</sub> ≈ W<sub>o</sub>x<sub>o</sub>', 'math')
p('x 表示输入特征，W 将特征映射到共同的关系空间，t 表示关系向量。相加的是映射后的数字表示；平移发生在学习到的特征空间。三种机制可以组合使用，并非固定的先后步骤。')
p('原文定位：第 3.1.1 至 3.1.6 节。本轮先掌握机制的作用，不要求推导网络或损失函数。', 'small')

# Page 4
newpage()
h('5 任务设置与评价指标')
p('数据集（Dataset）由图像及相应标注组成；人工参考答案称为真值（Ground Truth）。训练集用于学习，独立测试集用于评估。本文讨论 VRD、Visual Genome 等数据集。先确认任务设置，再比较分数。')
table(['任务', '评测时已提供的信息', '模型需要预测'], [
    ['PredCls 谓词分类', '真值框 B 和真值类别 O', '关系 R'],
    ['SGCls 场景图分类', '真值框 B', '类别 O 和关系 R'],
    ['SGGen 场景图生成', '仅输入图像', '框 B、类别 O 和关系 R'],
], [155, 170, WIDTH - 325])
h('R@K 真实关系找回了多少', 2)
p('对一张图，取分数最高的 K 条关系预测，计算其中匹配到的真值关系数占该图真值关系总数的比例。K 指关系预测条数，不是对象数量。')
story.append(RecallFraction())
p('例：真实关系有 5 条，模型前 10 条预测找回了其中 4 条，则 R@10 = 4/5 = 80%。分母是 5，不是 10。“前 10 条预测中有多大比例正确”对应精确率（Precision）的概念。')
h('mR@K 对关系类别等权平均', 2)
p('按同一评测协议取得各谓词类别的召回率，再在纳入评估的类别间等权平均。它衡量平均召回，不是模型预测某种关系的概率，也不是为每类重新挑选 K 条预测。')
table(['假设仅评估两类关系', '该类别的召回率'], [
    ['站在', '90%'],
    ['骑', '10%'],
], [315, WIDTH - 315])
p('在这个简化例子中，mR@K = (90% + 10%)/2 = 50%。各类别在平均中占相同分量，因此这个指标更有助于观察低频类别被忽略的问题。')
p('<b>满召回不等于图完全正确。</b>假设真值有 2 条，前 50 条预测包含全部真值，也可能含有许多错误预测，R@50 仍可达到 100%。错误预测虽然可能挤占前 K 名，召回率本身并不直接计算额外错误预测的比例。')
p('比较实验需对齐数据集及划分、任务、K、匹配标准和图约束。匹配关系时还需按任务要求核对对象类别与定位；跨图及类别内如何汇总，以对应评测协议为准。原文定位：第 4、5 节。', 'small')

# Page 5
newpage()
h('6 研究难点与全文复述')
table(['难点', '应怎样理解'], [
    ['有效关系稀疏', '候选对象对多，有意义的关系少。需要筛选值得判断的对象对。'],
    ['长尾分布', '少数关系类别或三元组样本很多，许多其他类别样本很少。关注训练样本分布不均。'],
    ['同类关系外观多样', '例如人骑马与人骑自行车都包含“骑”，视觉外观与交互细节却不同。'],
    ['关系定义和标注歧义', '“马在草地上”和“马站在草地上”可能都合理；标注粒度和词语含义可能不一致。'],
    ['评价不够全面', '单个召回指标无法完整反映错误预测、描述细致程度和整体场景图质量。'],
], [136, WIDTH - 136])
p('<b>区分两类数据问题：</b>减少无关对象对，主要应对有效关系稀疏与计算成本；改善低频关系的训练，主要应对长尾分布。有效关系少，与关系类别本身“单一”，是不同概念。')
h('视频与三维场景', 2)
p('<b>视频 SGG</b>需要描述关系随时间的变化，例如同一个人先牵马、后来骑马，并在不同帧中持续识别相同对象。<b>3D SGG</b>利用三维形状和位置；“前面、后面”等关系还需要明确参考方向。')
p('从该版本的视角，研究机会包括更准确且细粒度的数据标注、按具体场景设计模型，以及发展统一有效的三维场景图表示。原文定位：第 3.2、3.3、6 节。')
h('六句话复述整篇论文', 2)
for txt in [
    '1　SGG 从图像预测对象及其关系，并生成结构化的场景图。',
    '2　基本流程是定位对象、判断类别、预测关系，并用三元组组织结果。',
    '3　模型结合外观、空间、语义和上下文，也可以利用先验与常识。',
    '4　消息传递、注意力和视觉平移嵌入用于改善表示或关系判断，并且可以组合。',
    '5　主要困难包括关系稀疏、长尾分布、视觉差异、标注歧义和评测局限。',
    '6　评估要区分任务条件，同时结合整体召回与类别平均召回理解模型表现。',
]:
    p(txt)
p('回读路径：第 2 节定义 → 第 3 节流程与方法 → 第 5 节任务和指标 → 第 6 节挑战。沿“输入输出、方法作用、评测条件、剩余问题”四条线索做笔记。', 'small')

# Page 6
newpage()
h('复习检查')
p('先独立作答，再核对答案要点。能解释理由，比只记住英文术语更重要。')
questions = [
    'SGG 的输入和输出分别是什么？请写出〈人，骑，马〉之外的一条关系。',
    'B、O、R 分别回答什么问题？T 是否必须对应某个固定英文缩写？',
    '人位于马的上方属于哪类信息？为什么还不能据此确定“骑”？',
    '汇总周围对象的信息，并让其中一些信息占更大权重，结合了哪两种机制？',
    '视觉平移发生在图片坐标中，还是学习到的特征空间中？',
    '给出真值框和真值类别，只预测关系，属于哪种评测任务？',
    '真值有 5 条，前 10 条预测找回 4 条，R@10 是多少？分母是什么？',
    'mR@K 衡量关系预测概率吗？它如何对不同关系类别进行平均？',
    '为什么 R@50 达到 100% 仍不能证明整个场景图完全正确？',
    'A 多数对象对没有有意义关系；B 少数类别样本很多而多数类别很少。各是哪种难点？',
]
for i, q in enumerate(questions, 1):
    p(f'{i}　{q}', 'quiz')
h('答案要点', 2)
answers = [
    '1　输入图像，输出预测的对象与关系结构；例如〈马，站在，草地〉。',
    '2　B 定位，O 分类，R 描述联系；T 按作者定义表示场景图。',
    '3　空间信息；还需结合外观、姿态、交互区域和上下文等证据。',
    '4　消息传递和注意力。　5　学习到的特征空间。　6　PredCls。',
    '7　4/5 = 80%，分母是真值关系总数。',
    '8　衡量召回；先按协议计算类别召回率，再进行类别等权平均。',
    '9　全部真值可以被找回，同时仍存在额外的错误预测。',
    '10　A 是有效关系稀疏；B 是长尾分布。',
]
for a in answers:
    p(a, 'quiz')
h('来源与版本', 2)
p('Zhu G, Zhang L, Jiang Y, et al. Scene Graph Generation: A Comprehensive Survey. arXiv:2201.00443v2, 2022.', 'small')
p('<link href="https://arxiv.org/abs/2201.00443v2" color="#254F6B">原文 https://arxiv.org/abs/2201.00443v2</link>', 'small')
p('Zotero 条目 6YMYISGT。学习示例中的数值用于说明计算；后续论文的具体标签、匹配和汇总规则，需查相应数据集与评测协议。', 'small')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('CN', 8.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(52, 28, 'SGG综述阅读笔记  2022年v2版本')
    canvas.drawRightString(letter[0] - 52, 28, f'{doc.page:02d}')
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                        leftMargin=52, rightMargin=52,
                        topMargin=43, bottomMargin=45,
                        title='SGG综述中文阅读笔记',
                        author='学习笔记', subject='场景图生成综述的六轮中文入门阅读')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(str(OUT))
texts = [page.extract_text() or '' for page in reader.pages]
print(f'OUTPUT {OUT}')
print(f'PAGES {len(reader.pages)}')
print(f'CHARS {[len(t) for t in texts]}')
for i, txt in enumerate(texts, 1):
    print(f'PAGE {i}: {txt[:70]!r}')
assert len(reader.pages) == 6, 'Unexpected pagination; review and adjust content before delivery.'
assert all(len(t) > 300 for t in texts), 'Sparse or blank page.'
assert '4/5 = 80%' in '\n'.join(texts)
assert '长尾分布' in '\n'.join(texts)
