import codecs
import json
import os

from tqdm import tqdm


def read_file(path):
    with codecs.open(path, encoding="iso-8859-1") as f:
        content = f.readlines()

    data = {}

    parse_data = False

    parsed_keys = None
    parsed_values = None

    for line in content:
        if line.startswith("Patienten-ID"):
            data["ID"] = line.strip("Patienten-ID:\t").strip("\r\n")
            continue

        if line.startswith("-----"):
            parse_data = True
            continue

        if line.startswith("LINE"):
            parse_data = False
            continue

        if parse_data and parsed_keys is None:
            if line.startswith("PatientenName"):
                parsed_keys = line.strip("\r\n").split(";")
            else:
                parse_data + False
            continue

        if parse_data and parsed_values is None and parsed_keys is not None:
            parsed_values = (
                line.strip("\r\n")
                .replace(",", ".")
                .replace("Links", "Left")
                .replace("Rechts", "Right")
                .split(";")
            )

            curr_data = dict(zip(parsed_keys, parsed_values))

            for key in [
                "PatientenName",
                "PatientenID",
                "Geburtstag",
                "Datum",
                "Alter/Jahre",
                "Geschlecht",
                "Datenquelle",
                "",
            ]:
                curr_data.pop(key, None)

            for k, v in curr_data.items():
                if "°" in v:
                    curr_data[k] = v.strip("°")

            side = curr_data.pop("Side")

            data[side] = curr_data

            for k, v in curr_data.items():
                if v:
                    curr_data[k] = float(v)
                else:
                    curr_data[k] = None

            parsed_keys = None
            parse_data = False
            parsed_values = None

    return data


if __name__ == "__main__":
    root_dir = ""
    out_dir = ""

    os.makedirs(out_dir, exist_ok=True)

    for item in tqdm([os.path.join(root_dir, item) for item in os.listdir(root_dir)]):
        if os.path.isfile(item) and item.endswith(".txt"):
            data = read_file(item)
            with open(
                str(item).replace(root_dir, out_dir).replace(".txt", ".json"), "w"
            ) as f:
                json.dump(data, f, indent=4, sort_keys=True)
