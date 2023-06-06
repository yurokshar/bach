from __future__ import annotations
import dataclasses
import functools
import typing
from collections import defaultdict

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
)
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap

from tkinter import (
    Tk,
    Frame,
    Label,
    Entry,
    Button,
    StringVar,
    filedialog,
)
from tkinter.ttk import (
    Combobox,
)
from PIL import (
    ImageTk,
    Image,
)
import numpy


from src.config import settings
from src.utils import Matrix
from src.transforms import walsh_hadamard_image_transform
from src.hadamard import HadamardMatrix


@dataclasses.dataclass
class State:
    crop_regions: dict[tuple[int, int], bool]
    crop_buttons: dict[tuple[int, int], Button]
    coeff_inputs: dict[tuple[int, int], Entry]

    plots: dict

    source_image_path: str | None = None
    source_image: Image.Image | None = None
    source_image_label: Label | None = None

    target_image_label: Label | None = None

    transformation_func: typing.Callable[[State], Image.Image] | None = None

    original_spectrum_label: Label | None = None
    transformed_spectrum_label: Label | None = None

    quantization_input: Entry | None = None
    spectrum_upper_threshold: Entry | None = None
    spectrum_lower_threshold: Entry | None = None

    matrix_select: Combobox | None = None
    matrix_order: Combobox | None = None


def select_image(state: State):
    file_path = filedialog.askopenfilename(filetypes=[
        ("Image Files", settings.ENABLED_IMAGE_FILETYPES)
    ])

    if file_path:
        state.source_image_path = file_path
        state.source_image = Image.open(file_path)

    if state.source_image and state.source_image_label:
        image = state.source_image.copy()
        img = ImageTk.PhotoImage(
            image.resize(
                size=(settings.IMAGE_FRAME_SIZE, settings.IMAGE_FRAME_SIZE),
                resample=Image.Resampling.NEAREST
            ),
        )
        state.source_image_label.config(image=img)
        setattr(state.source_image_label, 'image', img)


def transform_image(state: State):
    if not state.source_image:
        return

    if state.target_image_label is None:
        return

    if state.transformation_func is None:
        return

    image = state.transformation_func(state)
    img = ImageTk.PhotoImage(
        image.resize(
            size=(settings.IMAGE_FRAME_SIZE, settings.IMAGE_FRAME_SIZE),
            resample=Image.Resampling.NEAREST
        ),
    )
    state.target_image_label.config(image=img)
    setattr(state.target_image_label, 'image', img)


def change_crop_state(state: State, i: int, j: int):
    state.crop_buttons[i, j].configure(bg = 'green' if state.crop_regions[i, j] else 'red')
    state.crop_regions[i, j] = not state.crop_regions[i, j]


def show_spectrum(state: State, matrix: Matrix, parent: Label):

    if parent in state.plots:
        ax = state.plots[parent]['ax']
        cb = state.plots[parent]['cb']
        fig = state.plots[parent]['fig']
        cb.remove()
        ax.clear()
        ax.remove()
        fig.clf()
        state.plots[parent]['canvas'].get_tk_widget().destroy()

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()  # type: ignore
    cmap=LinearSegmentedColormap.from_list('rwb', ['r', 'w', 'b'], N=256) 
    img = ax.matshow(
        matrix,
        cmap=cmap, 
        norm=SymLogNorm(
            vmin=-1.0,
            vmax=1.0,
            linthresh=0.01,  # type: ignore
            linscale=1.0,  # type: ignore
        ),
    ) 
    cb = fig.colorbar(img)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw_idle()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    state.plots[parent] = {
        'canvas': canvas,
        'ax': ax,
        'cb': cb,
        'fig': fig,
    }


def crop_filter(
    state: State,
    spectrum: Matrix,
    transformer: HadamardMatrix,
) -> Matrix:
    step = spectrum.shape[0] // 4

    for (i, j), cropped in state.crop_regions.items():
        if not cropped:
            continue
        spectrum[i*step:(i+1)*step, j*step:(j+1)*step] = 0

    return spectrum


def coeff_filter(
    state: State,
    spectrum: Matrix,
    transformer: HadamardMatrix,
) -> Matrix:
    step = spectrum.shape[0] // 4

    base_spectrum = spectrum
    spectrum = spectrum.copy()
    for (i, j), input in state.coeff_inputs.items():
        try:
            spectrum[i*step:(i+1)*step, j*step:(j+1)*step] *= float(input.get())
        except ValueError:
            return base_spectrum

    return spectrum


def lowhigh_filter(
    state: State,
    spectrum: Matrix,
    transformer: HadamardMatrix,
) -> Matrix:
    try:
        lower_threshold = None
        if state.spectrum_lower_threshold is not None:
            lower_threshold = float(state.spectrum_lower_threshold.get())
        upper_threshold = None
        if state.spectrum_upper_threshold is not None:
            upper_threshold = float(state.spectrum_upper_threshold.get())
    except ValueError:
        return spectrum

    if lower_threshold is not None:
        spectrum[spectrum < lower_threshold] = lower_threshold
    if upper_threshold is not None:
        spectrum[spectrum > upper_threshold] = upper_threshold
    return spectrum


def quantization_filter(state, spectrum, transformer, quantization: float = 10.0):
    # return numpy.floor_divide(spectrum, quantization) * quantization
    try:
        quantization = float(state.quantization_input.get())
    except ValueError:
        return spectrum
    return numpy.round(spectrum / quantization, 1) * quantization


def sharpening_filter(spectrum, transformer):
    spectrum[32:, 32:] = 0

    H = transformer.data
    img_blur = (H.T @ spectrum @ H) / (512*512)

    blur_spectrum = H @ img_blur @ H.T
    diff_spectrum = spectrum - blur_spectrum
    sharpened_spectrum = spectrum + 0.5 * diff_spectrum

    return sharpened_spectrum


def squared_filter(spectrum, transformer):
    for i in range(spectrum.shape[0]):
        for j in range(spectrum.shape[1]):
            val = spectrum[i][j]
            if numpy.abs(val) > 1.0:
                spectrum[i][j] = numpy.sqrt(numpy.abs(val)) * numpy.sign(val)
    return spectrum


def hadamard_transform(state: State):
    assert state.source_image
    assert state.original_spectrum_label
    assert state.transformed_spectrum_label

    filters = [
        functools.partial(crop_filter, state),
        functools.partial(coeff_filter, state),
        functools.partial(quantization_filter, state),
        functools.partial(lowhigh_filter, state),
        # squared_filter,
    ]
    
    text_to_strategy = {
        'Построение Сильвестра': HadamardMatrix.GenStrategy.sylvester,
        'Построение из matrix16_1': HadamardMatrix.GenStrategy.matrix16_1,
        'Построение из matrix32_1': HadamardMatrix.GenStrategy.matrix32_1,
    }

    strategy = HadamardMatrix.GenStrategy.default
    if state.matrix_select is not None:
        strategy = getattr(
            HadamardMatrix.GenStrategy,
            text_to_strategy.get(state.matrix_select.get(), 'default'),
            strategy,
        )

    print('strategy: ', strategy)
    transformer = HadamardMatrix(
        order=state.source_image.size[0],
        strategy=strategy,
    )

    if state.matrix_order is not None:
        order = state.matrix_order.get()
        if order == 'Порядок Адамара':
            pass
        elif order == 'Порядок Уолша':
            print('TO WALSH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            transformer.convert_to_walsh()

    orig_spectrum, spectrum, image = walsh_hadamard_image_transform(
        transformer=transformer,
        target=state.source_image,
        filters=filters,
    )

    show_spectrum(state, orig_spectrum, state.original_spectrum_label)
    show_spectrum(state, spectrum, state.transformed_spectrum_label)

    return image


def run():
    state = State(
        transformation_func=hadamard_transform,
        crop_regions=defaultdict(bool),
        crop_buttons=dict(),
        coeff_inputs=dict(),
        plots=dict(),
    )

    root = Tk()
    root.title(
        string=settings.MAIN_WINDOW_TITLE,
    )

    frame_display = Frame(
        root,
        width=2*settings.IMAGE_FRAME_SIZE + 64,
        height=2*settings.IMAGE_FRAME_SIZE + 32,
    )
    frame_display.pack_propagate(False)
    frame_display.pack(side='left', fill='both', expand=True)

    # FRAME IMAGES
    frame_images = Frame(
        frame_display, bg='yellow', relief='solid',
    )
    frame_images.pack_propagate(False)
    frame_images.pack(side='left', fill='both', padx=10, pady=5, expand=True)

    frame1 = Frame(
        frame_images,
        width=settings.IMAGE_FRAME_SIZE,
        height=settings.IMAGE_FRAME_SIZE,
        bg='lightblue',
        bd=5,
        relief='sunken',
    )
    frame1.pack_propagate(False)
    frame1.pack(side='top')

    state.source_image_label = Label(frame1)
    state.source_image_label.pack(fill='both', expand=True)

    frame2 = Frame(
        frame_images,
        width=settings.IMAGE_FRAME_SIZE,
        height=settings.IMAGE_FRAME_SIZE,
        bg='lightblue',
        bd=5,
        relief='sunken',
    )
    frame2.pack_propagate(False)
    frame2.pack(side='top')

    state.target_image_label = Label(frame2)
    state.target_image_label.pack(fill='both', expand=True)

    # FRAME SPECTRUM
    frame_spectrum = Frame(
        frame_display, bg='green', relief='solid',
    )
    frame_spectrum.pack_propagate(False)
    frame_spectrum.pack(side='right', fill='both', padx=10, pady=5, expand=True)

    frame3 = Frame(
        frame_spectrum,
        width=settings.IMAGE_FRAME_SIZE,
        height=settings.IMAGE_FRAME_SIZE,
        bg='lightblue',
        bd=5,
        relief='sunken',
    )
    frame3.pack_propagate(False)
    frame3.pack(side='top')

    state.original_spectrum_label = Label(frame3)
    state.original_spectrum_label.pack(fill='both', expand=True)

    frame4 = Frame(
        frame_spectrum,
        width=settings.IMAGE_FRAME_SIZE,
        height=settings.IMAGE_FRAME_SIZE,
        bg='lightblue',
        bd=5,
        relief='sunken',
    )
    frame4.pack_propagate(False)
    frame4.pack(side='top')

    state.transformed_spectrum_label = Label(frame4)
    state.transformed_spectrum_label.pack(fill='both', expand=True)

    controls = Frame(root)
    controls.pack(anchor='n', side='bottom', fill='both', expand=True, ipady=20)
    button1 = Button(
        controls,
        text="Выбрать изображение",
        command=functools.partial(select_image, state),
        font=('Helvetica', 16),
        bg="lightgreen",
        bd=3,
        activebackground="green",
        justify='center',
    )
    button1.grid(
        row=0,
        column=0,
        columnspan=2,
        padx=5,
        pady=5,
        sticky='nsew',
    )
    button2 = Button(
        controls,
        text="Применить трансформацию",
        command=functools.partial(transform_image, state),
        font=('Helvetica', 16),
        bg="lightgreen",
        bd=3,
        activebackground="green",
        justify='center',
    )
    button2.grid(
        row=1,
        column=0,
        columnspan=2,
        padx=5,
        pady=5,
        sticky='nsew',
    )

    crop_label = Label(
        controls,
        font=('Helvetica', 12),
        text='Вырезанные участки',
    )
    crop_label.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')
    crop_grid = Frame(controls)
    crop_grid.grid(row=2, column=1)

    for i in range(4):
        for j in range(4):
            button = Button(
                crop_grid,
                text=f'{i*4 + j + 1:>2}',
                command=functools.partial(change_crop_state, state, i, j),
                font=('Helvetica', 12),
                bg="green",
                bd=3,
                activebackground="green",
                # justify='center',
            )
            button.grid(row=i, column=j, sticky='nsew')
            state.crop_buttons[i, j] = button

    coeff_label = Label(
        controls,
        font=('Helvetica', 12),
        text='Коэффициенты',
    )
    coeff_label.grid(row=3, column=0, padx=5, pady=5, sticky='nsew')
    coeff_grid = Frame(controls)
    coeff_grid.grid(row=3, column=1)

    for i in range(4):
        for j in range(4):
            entry = Entry(
                coeff_grid,
                font=('Helvetica', 12),
                bd=3,
                width=3,
                textvariable=StringVar(value='1.0'),
            )
            entry.grid(row=i, column=j)
            state.coeff_inputs[i, j] = entry

    quantization_label = Label(
        controls,
        font=('Helvetica', 12),
        text='Коэффициент квантования',
    )
    quantization_label.grid(row=4, column=0, padx=5, pady=5, sticky='nsew')
    state.quantization_input = Entry(
        controls,
        font=('Helvetica', 12),
        bd=3,
    )
    state.quantization_input.grid(row=4, column=1, padx=5, pady=5, sticky='nsew')

    lowhigh_label = Label(
        controls,
        font=('Helvetica', 12),
        text='Граница отсечения (от/до)',
    )
    lowhigh_label.grid(row=5, column=0, padx=5, pady=5, sticky='nsew')
    lowhigh_frame = Frame(controls)
    lowhigh_frame.grid(row=5, column=1, padx=5, pady=5, sticky='nsew')

    state.spectrum_lower_threshold = Entry(
        lowhigh_frame,
        font=('Helvetica', 12),
        bd=3,
        width=10,
    )
    state.spectrum_lower_threshold.grid(row=0, column=0)

    state.spectrum_upper_threshold = Entry(
        lowhigh_frame,
        font=('Helvetica', 12),
        bd=3,
        width=10,
    )
    state.spectrum_upper_threshold.grid(row=0, column=1)

    matrix_label = Label(
        controls,
        font=('Helvetica', 12),
        text='Матрица Адамара',
    )
    matrix_label.grid(row=6, column=0, padx=5, pady=5, sticky='nsew')

    state.matrix_select = Combobox(
        controls,
        values=[
            'Построение Сильвестра',
            'Построение из matrix16_1',
            'Построение из matrix32_1',
        ],
    )
    state.matrix_select.current(0)
    state.matrix_select.grid(row=6, column=1, padx=5, pady=5, sticky='nsew')

    matrix_order = Label(
        controls,
        font=('Helvetica', 12),
        text='Упорядочивание матрицы',
    )
    matrix_order.grid(row=7, column=0, padx=5, pady=5, sticky='nsew')

    state.matrix_order = Combobox(
        controls,
        values=[
            'Порядок Адамара',
            'Порядок Уолша',
        ],
    )
    state.matrix_order.current(0)
    state.matrix_order.grid(row=7, column=1, padx=5, pady=5, sticky='nsew')

    root.mainloop()
