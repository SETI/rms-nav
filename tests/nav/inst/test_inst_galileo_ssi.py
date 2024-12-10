import nav.inst.inst_galileo_ssi as instgossi


def test_galileo_ssi_basic():
    url = 'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0017/G1/IO/C0349673965R.IMG'
    inst = instgossi.InstGalileoSSI.from_file(url)
    assert inst.obs.midtime == -110923771.01052806
