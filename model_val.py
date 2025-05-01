from ultralytics import YOLO

if __name__ == "__main__":
    model_path = r"C:\Users\hagmmart\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\Train1\weights\05387.041_KI-Modell.pt"
    data_path = r"C:\Users\hagmmart\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\4_Splitted_Train\data - DELL and Workstation - flex.yaml"

    model = YOLO(model_path)

    ergebnisse = model.val(data=data_path)

    # Ergebnisse anzeigen:
    print(ergebnisse.results_dict)
