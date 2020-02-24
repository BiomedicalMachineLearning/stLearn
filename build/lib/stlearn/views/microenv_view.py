from itertools import cycle
from PIL import ImageTk, Image
import sys
if sys.version_info[0] < 3:
    import Tkinter as tk     ## Python 2.x
else:
    import tkinter as tk     ## Python 3.x
from tkinter.filedialog import asksaveasfile 
from tkinter import messagebox, Menu
from anndata import AnnData
from stlearn._compat import Literal
from typing import Optional
import matplotlib.pyplot as plt
from tkinter import PhotoImage
class ChangeImage():
    def __init__(self, root, adata, use_data):
        self.adata = adata
        self.use_data=use_data
        self.pil_imgs = []
        self.photos=[]
        self.routes = []
        self.load_images(root)
        self.image_num=0
        self.canvas = tk.Canvas(root, width=self.pil_imgs[0].size[0], 
                                    height=self.pil_imgs[0].size[1],bg="#fff",bd=0)  
        self.canvas.pack(fill=tk.BOTH,expand=True,side=tk.TOP)
        self.canvas.bind('<Configure>', self._resize_image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photos[self.image_num])
        ## click on button to change image
        self.bt_plot = tk.Button(root, text="Plot!", 
                                 command=lambda:[plot(self.adata,use_data,self.image_num+1),root.destroy()],
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_plot.pack(side=tk.RIGHT, padx=5, expand=True)
        menu = Menu(root,bg="#fff",bd=0)
        root.config(menu=menu,width = 20,bg="#fff",bd=0)
        filemenu = Menu(menu)
        menu.add_cascade(label="  File  ", menu=filemenu)
        filemenu.add_command(label="  Save  ", command=self.save)
        filemenu.add_separator()
        filemenu.add_command(label="  Exit  ", command=root.destroy)
        #self.bt_save = tk.Button(root, text="Save", command=self.save)
        #self.bt_save.pack(side=tk.RIGHT,expand=True)
        self.bt_next = tk.Button(root, text="Next", command=self.next_image,
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_next.pack(side=tk.RIGHT,expand=True)
        self.bt_next.config( height = 1, width = 5 )
        self.bt_prev = tk.Button(root, text="Prev", command=self.previous_image,
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_prev.pack(side=tk.LEFT,expand=True)
        self.bt_prev.config( height = 1, width = 5 )
        self.bt_select = tk.Button(root, text="Select", command=self.select_images,
                                  bg="#007bff",activebackground="#0053ac",foreground="#fff",
                                  activeforeground="#fff",bd=0)
        self.bt_select.pack(fill=tk.X, padx=5, expand=True)
        self.bt_select.config( height = 1, width = 5 )
        self.frame1 = tk.Frame(root,bg="#fff",bd=0)
        self.frame1.pack(fill=tk.X)
        self.lbl1 = tk.Label(self.frame1, text=("Factor: "), width=8,bg="#fff")
        self.lbl1.pack(side=tk.LEFT, padx=5, pady=5)
        self.v = tk.StringVar()
        self.v.set(str(self.image_num + 1))
        self.entry1 = tk.Entry(self.frame1,textvariable=self.v,bg="#dedede",bd=0)
        self.entry1.bind('<Return>', self.select_images)
        self.entry1.pack(fill=tk.X, padx=5, expand=True)
        #self.bt_exit = tk.Button(root, text="Exit", bg="orange", command=root.destroy)
        #self.bt_exit.pack(side=tk.LEFT)
    def next_image(self):
        self.image_num += 1
        if self.image_num >= len(self.photos):
            self.image_num=0
        ## pipe the next image to be displayed to the button
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)

        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
        self.v.set(str(self.image_num + 1))
    def previous_image(self):
        self.image_num = self.image_num - 1
        if self.image_num < 0:
            self.image_num=len(self.photos)-1
        ## pipe the next image to be displayed to the button
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
        self.v.set(str(self.image_num + 1))
    def load_images(self,root):
        """ copy data images to a list that is an instance variable
            all images are hard-coded here and so use data=
            instead of file=
        """
        ## put the images in an instance object (self.) so they aren't destroyed
        ## when the function exits and can be used anywhere in the class
        for item in self.adata.uns['plots'][self.use_data].items():
            self.pil_imgs.append(Image.fromarray(item[1]))
            self.photos.append(ImageTk.PhotoImage(image=Image.fromarray(item[1]),master=root))
    def select_images(self,*args):
        if ((int(self.v.get()) > len(self.photos)) or (int(self.v.get()) <= 0 )):
            messagebox.showerror("Error","Please choose available factor! (0 < factor <= " + str(len(self.photos)) + ")")
            return None
        self.image_num = int(self.v.get())-1
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
    def save(self): 
        f = asksaveasfile(filetypes=(("Portable Network Graphics (*.png)", "*.png"),
                                        ("All Files (*.*)", "*.*")),
                             mode='wb',
                             defaultextension='.png')
        if f is None:
            return
        filename = f.name
        extension = filename.rsplit('.', 1)[-1]
        img = Image.fromarray(self.adata.uns['plots'][self.use_data]["factor_" + str(self.image_num+1)])
        img.save(f, extension)
        f.close()
    def _resize_image(self,event):
        self.new_width = event.width
        self.new_height = event.height
        self.image = self.pil_imgs[self.image_num].resize((self.new_width, self.new_height))
        self.image = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.image_on_canvas, image =self.image)
def microenv_plot(
    adata: AnnData,
    use_data: str = "fa",
    factor: int = None,
    dpi: int = 192,):
    if use_data not in ["X_fa","X_ica","X_ldvae"]:
        raise ValueError("Please provide one of these data: X_fa, X_ica or X_ldvae")
    if use_data not in str(adata.obsm):
        raise ValueError("Please run function: run_" + use_data )
    if factor is not None:
        plot(adata,use_data,factor)
    else:
        root=tk.Tk()
        root.title('stLearn - Factor Analysis')
        root.configure(background='white')
        CI=ChangeImage(root,adata,use_data)
        root.mainloop()
def plot(
    adata: AnnData,
    use_data: str = "X_fa",
    factor: int = None,
    dpi: int = 192,
    copy: bool = False,
) -> Optional[AnnData]:
    if ((factor > len(adata.uns['plots'][use_data])) or (factor <= 0 )):
        raise ValueError("Error","Please choose available factor! (0 < factor <= " + str(len(adata.uns['plots'][use_data])) + ")") 
    else:
        plt.rcParams['figure.dpi'] = dpi
        plt.imshow(adata.uns['plots'][use_data]["factor_"+str(factor)])
        plt.axis('off')
        plt.show()