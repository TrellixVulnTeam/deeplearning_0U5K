from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_ip(domain):
    import requests
    from bs4 import BeautifulSoup
    url = "http://ip.chinaz.com/" + domain
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text)
    x = soup.find(class_="IcpMain02")

    x = x.find_all("span", class_="Whwtdhalf")
    try:
        print("%s %s" % (x[5].text, x[4].text))
    except:
        pass


get_available_gpus()
device_lib.list_local_devices()