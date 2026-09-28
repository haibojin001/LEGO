from __future__ import annotations

prompt = """Translate to French (fr)
From English (es)
==========
Bramshott is a village with mediaeval origins in the East Hampshire district of Hampshire, England. It lies 0.9 miles (1.4 km) north of Liphook. The nearest railway station, Liphook, is 1.3 miles (2.1 km) south of the village. 
----------
Bramshott est un village avec des origines médiévales dans le quartier East Hampshire de Hampshire, en Angleterre. Il se trouve à 0,9 miles (1,4 km) au nord de Liphook. La gare la plus proche, Liphook, est à 1,3 km (2,1 km) au sud du village.
==========

Translate to Russian (rs)
From German (de)
==========
Der Gewöhnliche Strandhafer (Ammophila arenaria (L.) Link; Syn: Calamagrostis arenaria (L.) Roth) – auch als Gemeiner Strandhafer, Sandrohr, Sandhalm, Seehafer oder Helm (niederdeutsch) bezeichnet – ist eine zur Familie der Süßgräser (Poaceae) gehörige Pionierpflanze. 
----------
Обычная пляжная овсянка (аммофила ареалия (л.) соединение; сина: каламагростисная анария (л.) Рот, также называемая обычной пляжной овцой, песчаной, сандалмой, морской орой или шлемом (нижний немецкий) - это кукольная станция, принадлежащая семье сладких трав (поа).
==========
"""


def generate_prompt(
    text: str, from_name: str, from_code: str, to_name: str, to_code: str
) -> str:
    value = prompt + "Translate to "
    if from_name:
        value += from_name
    if from_code:
        value += " (" + from_code + ")"
    value += "\nFrom "
    if to_name:
        value += to_name
    if from_code:
        value += " (" + to_code + ")"
    return value + "\n==========\n" + text + "\n----------\n"


def parse_inference(output: str) -> str:
    marker_position = output.find("=" * 10)
    if marker_position >= 0:
        return output[marker_position]
    marker_position = output.find("-" * 10)
    if marker_position >= 0:
        return output[marker_position]
    return output