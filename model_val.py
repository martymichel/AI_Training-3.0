from ultralytics import YOLO

if __name__ == "__main__":
    model_path = r"C:\Users\miche\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\Train1\weights\best.pt"
    data_path = r"C:\Users\miche\OneDrive - Flex\3_ARBEIT ARBEIT ARBEIT\5.2_AI_Camera tests\M_inner_B_Reklamation_3AA01995\4_Splitted_Train\data - ASUS.yaml"

    model = YOLO(model_path)

    ergebnisse = model.val(data=data_path)

    # Ergebnisse anzeigen:
    print(ergebnisse.results_dict)
