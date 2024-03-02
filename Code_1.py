#pip install pillow
#j'ai du mettre à jour le pip pour que ça marche

from PIL import Image

i = Image.open("0.JPG")
#i.show()

n = 0
for x in range(0, i.width):
    for y in range(0, i.height):
        c = i.getpixel((x, y))
        print(c)
        n = n + 1
# juste pour vérifier qu'il y en a bien 100 pixels

