"""The tray icon (pystray): open the page, open the data folder, close. The page does everything else.

If there is no tray (pystray or Pillow missing, no desktop session, or pystray fails), run_tray() returns False and
the launcher keeps running without an icon (the page still works; __main__ opens it in the browser once).
"""
import threading


def make_icon_image(size=64):
    """A red disc with "MV" (RD Tests red), drawn with Pillow: no image file to ship."""
    from PIL import Image, ImageDraw
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((2, 2, size - 2, size - 2), fill=(193, 0, 50, 255))
    text = "MV"
    left, top, right, bottom = draw.textbbox((0, 0), text)
    draw.text(((size - (right - left)) / 2, (size - (bottom - top)) / 2 - top), text, fill="white")
    return image


def run_tray(launcher, url, open_browser, log=print):
    """Show the icon and block until the launcher should exit. Returns False if no tray could be shown."""
    try:
        import pystray
        image = make_icon_image()
    except Exception as e:  # noqa: BLE001 (ImportError, or no display backend on Linux)
        log("sem ícone na bandeja (%s: %s); o painel continua em %s" % (type(e).__name__, e, url))
        return False

    def open_page(icon=None, item=None):
        open_browser(url)

    def open_data(icon=None, item=None):
        try:
            launcher.open_folder(launcher.paths.data_root)
        except Exception as e:  # noqa: BLE001
            log("abrir pasta: %s" % e)

    def close_keep_running(icon, item):
        launcher.request_exit(stop_roles=False)

    def close_stop_all(icon, item):
        launcher.request_exit(stop_roles=True)

    menu = pystray.Menu(
        pystray.MenuItem("Abrir painel", open_page, default=True),
        pystray.MenuItem("Abrir pasta de dados", open_data),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Fechar o painel (os papéis continuam rodando)", close_keep_running),
        pystray.MenuItem("Parar tudo e sair", close_stop_all),
    )
    icon = pystray.Icon("bajamv", image, "Baja MV", menu)

    def stop_icon_on_exit():
        launcher.exit_requested.wait()
        icon.stop()

    threading.Thread(target=stop_icon_on_exit, daemon=True).start()
    try:
        icon.run()
    except Exception as e:  # noqa: BLE001
        log("ícone da bandeja falhou (%s: %s); o painel continua em %s" % (type(e).__name__, e, url))
        return False
    return True
