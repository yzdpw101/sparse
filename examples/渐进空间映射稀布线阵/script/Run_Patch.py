"""HFSS 贴片线阵仿真 — 可被 Python 直接调用的函数。

用法:
    from Run_Patch import run_patch_simulation
    run_patch_simulation(x_centers=[0, 0.06], frequency_ghz=2.45)
"""

import os, math


def run_patch_simulation(
    x_centers: list[float],
    y_centers: list[float] = None,
    magnitudes: list[float] = None,
    phases_deg: list[float] = None,
    frequency_ghz: float = 2.45,
    results_dir: str = "",
    run_simulation: bool = True,
    close_after: bool = False,
    project_save_path: str = "",
    # 贴片/基板参数
    epsilon_r: float = 4.4,
    patch_length_mm: float = 28.0,
    patch_height_mm: float = 0.035,
    substrate_height_mm: float = 1.6,
    array_width_wl: float = 0.0,
    array_length_wl: float = 0.0,
    array_margin_wl: float = 0.5,
    airbox_margin_wl: float = 0.25,
    gnd_height_mm: float = 0.03,
    port_radius_mm: float = 1.5,
    feed_radius_mm: float = 0.6,
    l1_mm: float = 6.7,
    # 求解器参数
    max_delta_s: float = 0.02,
    max_passes: int = 50,
    save_rad_fields_only: bool = True,
    # 远场辐射球参数
    theta_start: float = -180,
    theta_stop: float = 180,
    theta_step: float = 0.1,
    phi_start: float = 0,
    phi_stop: float = 360,
    phi_step: float = 1.0,
    # 远场导出参数
    results_category: str = "GainTotal",
    results_function: str = "",
    results_phi_sections: list[float] = None,
    excitation_type: str = "LumpedPort",
):
    """在 HFSS 中建模仿真贴片线阵，导出远场方向图 CSV。

    Args:
        x_centers: 阵元 x 坐标 (m)
        y_centers: 阵元 y 坐标 (m), None=全0
        magnitudes: 激励幅度, None=全1
        phases_deg: 激励相位 (度), None=全0
        frequency_ghz: 工作频率 (GHz)
        results_dir: 结果输出目录
        run_simulation: True=仿真并导出, False=仅建模
        close_after: 仿真后是否关闭 HFSS
    """

    num_elems = len(x_centers)
    if y_centers is None:
        y_centers = [0.0] * num_elems
    if magnitudes is None:
        magnitudes = [1.0] * num_elems
    if phases_deg is None:
        phases_deg = [0.0] * num_elems
    if results_phi_sections is None:
        results_phi_sections = []

    # 绝对路径
    results_dir = os.path.abspath(results_dir)
    xc_mm = [x * 1000 for x in x_centers]
    yc_mm = [y * 1000 for y in y_centers]
    light_speed = 299792458

    # ── 启动 HFSS ──
    import win32com.client
    oAnsoftApp = win32com.client.Dispatch("Ansoft.ElectronicsDesktop")
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()
    oProject = oDesktop.NewProject()
    oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Modal Network", "")
    oDesign = oProject.SetActiveDesign("HFSSDesign1")
    oEditor = oDesign.SetActiveEditor("3D Modeler")

    # ── 变量 ──
    oDesign.ChangeProperty(["NAME:AllTabs", ["NAME:LocalVariableTab",
        ["NAME:PropServers", "LocalVariables"],
        ["NAME:NewProps",
            mkvar("patchLength", f"{patch_length_mm}mm"),
            mkvar("frequency", f"{frequency_ghz}e9"),
            mkvar("lightSpeed", f"{light_speed}"),
            mkvar("wavelength", f"{light_speed / frequency_ghz / 1e9 * 1000}mm"),
            mkvar("epsilon_r", f"{epsilon_r}"),
            mkvar("portRadius", f"{port_radius_mm}mm"),
            mkvar("feedRadius", f"{feed_radius_mm}mm"),
            mkvar("L1", f"{l1_mm}mm"),
            mkvar("patchWidth",
                  f"{light_speed / (2 * frequency_ghz * 1e9) * math.sqrt(2 / (epsilon_r + 1)) * 1000}mm"),
            mkvar("substrateWidth", f"wavelength * {array_width_wl} + wavelength * {array_margin_wl}"),
            mkvar("substrateLength", f"wavelength * {array_length_wl} + wavelength * {array_margin_wl}"),
            mkvar("substrateHeight", f"{substrate_height_mm}mm"),
            mkvar("airBoxWidth", f"substrateWidth + wavelength * {airbox_margin_wl}"),
            mkvar("airBoxLength", f"substrateLength + wavelength * {airbox_margin_wl}"),
            mkvar("patchHeight", f"{patch_height_mm}mm"),
            mkvar("GNDHeight", f"{gnd_height_mm}mm"),
            mkvar("airBoxHeight",
                  f"substrateHeight + patchHeight + GNDHeight + wavelength * {airbox_margin_wl}"),
        ]]])

    # ── 基板 & 地 ──
    oEditor.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", "-substrateLength /2",
         "YPosition:=", "-substrateWidth / 2", "ZPosition:=", "0mm",
         "XSize:=", "substrateLength", "YSize:=", "substrateWidth",
         "ZSize:=", "substrateHeight"],
        mkattr("Substrate", "(0 128 0)", "FR4_epoxy", solve_inside=True))
    oEditor.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", "-substrateLength / 2",
         "YPosition:=", "-substrateWidth / 2", "ZPosition:=", "- GNDHeight",
         "XSize:=", "substrateLength", "YSize:=", "substrateWidth",
         "ZSize:=", "GNDHeight"],
        mkattr("GND", "(255 128 0)", "copper", solve_inside=False))

    # ── 阵元 ──
    print("Start modeling...")
    port_faces, port_holes, feed_centers = [], "", []
    for i in range(num_elems):
        idx = i + 1
        xc, yc = xc_mm[i], yc_mm[i]
        fx = f"{xc}mm + L1"

        oEditor.CreateBox(
            ["NAME:BoxParameters", "XPosition:=", f"{xc}mm - patchLength/2",
             "YPosition:=", f"{yc}mm - patchWidth / 2", "ZPosition:=", "substrateHeight",
             "XSize:=", "patchLength", "YSize:=", "patchWidth", "ZSize:=", "patchHeight"],
            mkattr(f"Patch{idx}", "(255 128 0)", "copper"))
        oEditor.CreateCylinder(
            ["NAME:CylinderParameters", "XCenter:=", fx, "YCenter:=", f"{yc}mm",
             "ZCenter:=", "-GNDHeight", "Radius:=", "feedRadius",
             "Height:=", "GNDHeight + substrateHeight", "WhichAxis:=", "Z", "NumSides:=", "0"],
            mkattr(f"Feed{idx}", "(255 128 0)", "copper"))
        oEditor.CreateCylinder(
            ["NAME:CylinderParameters", "XCenter:=", fx, "YCenter:=", f"{yc}mm",
             "ZCenter:=", "-GNDHeight", "Radius:=", "portRadius",
             "Height:=", "substrateHeight + GNDHeight", "WhichAxis:=", "Z", "NumSides:=", "0"],
            mkattr(f"PortHole{idx}", "(143 175 143)", "vacuum", solve_inside=True))
        oEditor.CreateCircle(
            ["NAME:CircleParameters", "IsCovered:=", True,
             "XCenter:=", fx, "YCenter:=", f"{yc}mm", "ZCenter:=", "-GNDHeight",
             "Radius:=", "portRadius", "WhichAxis:=", "Z", "NumSegments:=", "0"],
            mkattr(f"portFace{idx}", "(143 175 143)", "vacuum", solve_inside=True))

        port_faces.append(f"portFace{idx}")
        port_holes = f"PortHole{idx}" if i == 0 else port_holes + f", PortHole{idx}"
        feed_centers.append((xc + l1_mm, yc))

    # ── 布尔减 ──
    if port_holes:
        oEditor.Subtract(
            ["NAME:Selections", "Blank Parts:=", "Substrate, GND",
             "Tool Parts:=", port_holes],
            ["NAME:SubtractParameters", "KeepOriginals:=", False, "TurnOnNBodyBoolean:=", True])

    # ── 端口 ──
    oModule = oDesign.GetModule("BoundarySetup")
    for idx, pf in enumerate(port_faces, start=1):
        fx, fy = feed_centers[idx - 1]
        s = [f"{fx + feed_radius_mm}mm", f"{fy}mm", f"-{gnd_height_mm}mm"]
        e = [f"{fx + port_radius_mm}mm", f"{fy}mm", f"-{gnd_height_mm}mm"]
        if excitation_type == "Voltage":
            oModule.AssignVoltage(
                ["NAME:" + str(idx), "Objects:=", [pf],
                 ["NAME:Direction", "Coordinate System:=", "Global", "Start:=", s, "End:=", e]])
        else:
            oModule.AssignLumpedPort(
                ["NAME:" + str(idx), "Objects:=", [pf], "LumpedPortType:=", "Default",
                 "DoDeembed:=", False,
                 ["NAME:Modes", ["NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True,
                  ["NAME:IntLine", "Coordinate System:=", "Global", "Start:=", s, "End:=", e],
                  "AlignmentGroup:=", 0, "CharImp:=", "Zpi", "RenormImp:=", "50ohm"]],
                 "Impedance:=", "50ohm"])

    # ── 空气盒 + 辐射边界 ──
    oEditor.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", "-airBoxLength / 2",
         "YPosition:=", "-airBoxWidth / 2", "ZPosition:=", "-airBoxHeight / 2",
         "XSize:=", "airBoxLength", "YSize:=", "airBoxWidth", "ZSize:=", "airBoxHeight"],
        mkattr("AirBox", "(143 175 143)", "air", solve_inside=True, transparency=0.8))
    oModule.AssignRadiation(["NAME:Rad1", "Objects:=", ["AirBox"]])

    # ── 激励源 ──
    oModuleSolutions = oDesign.GetModule("Solutions")
    EditSources_args = [
        ["IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False]
    ] + [
        ["Name:=", f"{i+1}:1", "Magnitude:=", f"{magnitudes[i]}W",
         "Phase:=", f"{phases_deg[i]}deg"]
        for i in range(num_elems)
    ]
    oModuleSolutions.EditSources(EditSources_args)

    # ── 远场设置 ──
    sphere_name = "Infinite Sphere1"
    oModule = oDesign.GetModule("RadField")
    oModule.InsertInfiniteSphereSetup([
        f"NAME:{sphere_name}", "UseCustomRadiationSurface:=", False,
        "CSDefinition:=", "Theta-Phi", "Polarization:=", "Linear",
        "ThetaStart:=", f"{theta_start}deg", "ThetaStop:=", f"{theta_stop}deg",
        "ThetaStep:=", f"{theta_step}deg",
        "PhiStart:=", f"{phi_start}deg", "PhiStop:=", f"{phi_stop}deg",
        "PhiStep:=", f"{phi_step}deg", "UseLocalCS:=", False])

    # ── 求解设置 ──
    oModule = oDesign.GetModule("AnalysisSetup")
    oModule.InsertSetup("HfssDriven", [
        "NAME:Setup1", "SolveType:=", "Single", "Frequency:=", f"{frequency_ghz}GHz",
        "MaxDeltaS:=", max_delta_s, "UseMatrixConv:=", False,
        "MaximumPasses:=", max_passes, "MinimumPasses:=", 1, "MinimumConvergedPasses:=", 1,
        "PercentRefinement:=", 30, "IsEnabled:=", True,
        ["NAME:MeshLink", "ImportMesh:=", False],
        "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True,
        "SetLambdaTarget:=", False, "Target:=", 0.3333, "UseMaxTetIncrease:=", False,
        "PortAccuracy:=", 2, "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False,
        "DrivenSolverType:=", "Direct Solver", "EnhancedLowFreqAccuracy:=", False,
        "SaveRadFieldsOnly:=", save_rad_fields_only, "SaveAnyFields:=", True,
        "IESolverType:=", "Auto", "LambdaTargetForIESolver:=", 0.15,
        "UseDefaultLambdaTgtForIESolver:=", True, "IE Solver Accuracy:=", "Balanced",
        "InfiniteSphereSetup:=", sphere_name,
        "MaxPass:=", 10, "MinPass:=", 1, "MinConvPass:=", 1,
        "PerError:=", 1, "PerRefine:=", 30])

    if project_save_path:
        oProject.SaveAs(project_save_path, True)
    print("Modeling completed")

    # ── 仿真 + 导出 ──
    if not run_simulation:
        print("Simulation skipped.")
        return oDesign, oProject, oDesktop

    print("Starting simulation...")
    oDesign.Analyze("Setup1")
    print("Simulation completed.")

    oModule = oDesign.GetModule("ReportSetup")
    os.makedirs(results_dir, exist_ok=True)

    report_type = f"{results_function}({results_category})"
    phi_entries = "All" if not results_phi_sections else [f"{p}deg" for p in results_phi_sections]
    report_name = results_category
    export_path = os.path.join(results_dir, f"{report_type}.csv")

    oModule.CreateReport(report_name, "Far Fields", "Rectangular Plot",
        "Setup1 : LastAdaptive",
        ["Context:=", "Infinite Sphere1"],
        ["Theta:=", ["All"], "Phi:=", [phi_entries], "Freq:=", [f"{frequency_ghz}GHz"],
         "patchLength:=", ["Nominal"], "frequency:=", ["Nominal"],
         "lightSpeed:=", ["Nominal"], "epsilon_r:=", ["Nominal"],
         "portRadius:=", ["Nominal"], "feedRadius:=", ["Nominal"],
         "L1:=", ["Nominal"], "substrateHeight:=", ["Nominal"],
         "patchHeight:=", ["Nominal"], "GNDHeight:=", ["Nominal"]],
        ["X Component:=", "Theta", "Y Component:=", [report_type]])

    try:
        oModule.ExportUniformPointsToFile(report_name, export_path,
            f"{theta_start}deg", f"{theta_stop}deg",
            f"{theta_step}deg", True, "", False, True)
        print(f"Exported to {export_path}")
    except Exception as ex:
        print(f"Export error: {ex}")

    print("Simulation and export completed.")

    if close_after:
        oDesktop.CloseProject(oProject.GetName())
        oDesktop.QuitApplication()

    return oDesign, oProject, oDesktop


# ── 内部辅助 ──

def mkvar(name, value):
    return ["NAME:" + name, "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", value]

def mkattr(name, color, material, solve_inside=False, transparency=0):
    return ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", color,
            "Transparency:=", transparency, "PartCoordinateSystem:=", "Global",
            "UDMId:=", "", "MaterialValue:=", f'"{material}"',
            "SurfaceMaterialValue:=", '""', "SolveInside:=", solve_inside,
            "ShellElement:=", False, "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel", "IsMaterialEditable:=", True,
            "IsSurfaceMaterialEditable:=", True, "UseMaterialAppearance:=", False,
            "IsLightweight:=", False]


if __name__ == '__main__':
    import json as _json
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = _json.load(open(os.path.join(here, "Run_Patch_Config.json"),
                          encoding="utf-8"))["script"]["hfss"]
    xf = _json.load(open(os.path.join(here, "Xf.json"), encoding="utf-8"))

    out_dir = cfg["pattern_export_directory"] or here

    run_patch_simulation(
        x_centers=xf["xCenters"],
        y_centers=xf.get("yCenters"),
        magnitudes=xf.get("magnitudes"),
        phases_deg=xf.get("phasesDeg"),
        frequency_ghz=cfg["frequency_GHz"],
        results_dir=out_dir,
        run_simulation=cfg["run_simulation"],
        close_after=cfg["close_after_simulation"],
        project_save_path=cfg.get("project_save_path", ""),
        epsilon_r=cfg["epsilon_r"],
        patch_length_mm=cfg["patchLength_mm"],
        patch_height_mm=cfg["patchHeight_mm"],
        substrate_height_mm=cfg["substrateHeight_mm"],
        array_width_wl=cfg["arrayWidth_wavelength"],
        array_length_wl=cfg["arrayLength_wavelength"],
        array_margin_wl=cfg["arrayMargin_wavelength"],
        airbox_margin_wl=cfg["airboxMargin_wavelength"],
        gnd_height_mm=cfg["GNDHeight_mm"],
        port_radius_mm=cfg["portRadius_mm"],
        feed_radius_mm=cfg["feedRadius_mm"],
        l1_mm=cfg["L1_mm"],
        max_delta_s=cfg["InsertSetup_MaxDeltaS"],
        max_passes=cfg["InsertSetup_MaximumPasses"],
        save_rad_fields_only=cfg["InsertSetup_SaveRadFieldsOnly"],
        theta_start=cfg["ThetaStart_deg"],
        theta_stop=cfg["ThetaStop_deg"],
        theta_step=cfg["ThetaStep_deg"],
        phi_start=cfg["PhiStart_deg"],
        phi_stop=cfg["PhiStop_deg"],
        phi_step=cfg["PhiStep_deg"],
        results_category=cfg["Results_Category"],
        results_function=cfg.get("Results_Function", ""),
        results_phi_sections=cfg.get("Results_Sections_deg", []),
        excitation_type=cfg["Assign_Excitation_Type"],
    )


