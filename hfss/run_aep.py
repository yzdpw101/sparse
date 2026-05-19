"""HFSS 有源单元方向图 (AEP) 仿真 — 逐元激励导出。

用法:
    from Run_AEP import run_aep_simulation
    csv_paths = run_aep_simulation(x_centers=[0, 0.06], frequency_ghz=2.45, ...)

流程:
    1. 建立完整阵列几何 (与 Run_Patch.py 相同)
    2. 所有端口一次性仿真 (自适应网格收敛)
    3. 逐个端口激励: EditSources → 重新求解 → 导出远场 CSV
       - 第 i 个端口: 1W / 0°
       - 其余端口: 0W (等效 50Ω 匹配负载)
    4. 返回所有 AEP CSV 路径

输出文件命名: aep_elem_{i}_{freq}GHz.csv
    i=1: 最左侧单元, i=N: 最右侧单元 (按 x 坐标排序)
"""

import os, math


def run_aep_simulation(
    x_centers: list[float],
    y_centers: list[float] = None,
    frequency_ghz: float = 2.45,
    results_dir: str = "",
    close_after: bool = False,
    # 贴片/基板参数 (与 Run_Patch.py 一致)
    epsilon_r: float = 4.4,
    patch_length_mm: float = 27.9,
    patch_height_mm: float = 0.035,
    substrate_height_mm: float = 1.6,
    array_width_wl: float = 0.0,
    array_length_wl: float = 0.0,
    array_margin_wl: float = 0.5,
    airbox_margin_wl: float = 0.25,
    gnd_height_mm: float = 0.03,
    port_radius_mm: float = 1.5,
    feed_radius_mm: float = 0.6,
    l1_mm: float = 7.7,
    # 求解器参数
    max_delta_s: float = 0.02,
    max_passes: int = 50,
    save_rad_fields_only: bool = True,
    # 远场辐射球参数
    theta_start: float = -180,
    theta_stop: float = 180,
    theta_step: float = 0.1,
    phi_start: float = 0,
    phi_stop: float = 0,
    phi_step: float = 1,
    # 远场导出参数
    results_category: str = "GainTotal",
    results_function: str = "",
    results_phi_sections: list[float] = None,
    # AEP 特有参数
    aep_elements: list[int] = None,  # None=全部; [1,2,3] 仅指定单元
) -> list[str]:
    """AEP 仿真: 建模一次, 逐端口激励导出方向图。

    Args:
        x_centers: 阵元 x 坐标 (m), 已按 x 从左到右排序
        y_centers: 阵元 y 坐标 (m), None=全 0
        aep_elements: 要仿真的单元序号列表 (1-based), None=全部

    Returns:
        导出的 CSV 文件路径列表, 按单元序号排列
    """
    num_elems = len(x_centers)
    if y_centers is None:
        y_centers = [0.0] * num_elems
    if results_phi_sections is None:
        results_phi_sections = []

    results_dir = os.path.abspath(results_dir)
    xc_mm = [x * 1000 for x in x_centers]
    yc_mm = [y * 1000 for y in y_centers]
    light_speed = 299792458

    if aep_elements is None:
        aep_elements = list(range(1, num_elems + 1))

    # ══════════════════════════════════════════════════════
    # 1. 启动 HFSS + 建模 (与 Run_Patch.py 相同)
    # ══════════════════════════════════════════════════════
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
            _mkvar("patchLength", f"{patch_length_mm}mm"),
            _mkvar("frequency", f"{frequency_ghz}e9"),
            _mkvar("lightSpeed", f"{light_speed}"),
            _mkvar("wavelength", f"{light_speed / frequency_ghz / 1e9 * 1000}mm"),
            _mkvar("epsilon_r", f"{epsilon_r}"),
            _mkvar("portRadius", f"{port_radius_mm}mm"),
            _mkvar("feedRadius", f"{feed_radius_mm}mm"),
            _mkvar("L1", f"{l1_mm}mm"),
            _mkvar("patchWidth",
                  f"{light_speed / (2 * frequency_ghz * 1e9) * math.sqrt(2 / (epsilon_r + 1)) * 1000}mm"),
            _mkvar("substrateWidth", f"wavelength * {array_width_wl} + wavelength * {array_margin_wl}"),
            _mkvar("substrateLength", f"wavelength * {array_length_wl} + wavelength * {array_margin_wl}"),
            _mkvar("substrateHeight", f"{substrate_height_mm}mm"),
            _mkvar("airBoxWidth", f"substrateWidth + wavelength * {airbox_margin_wl}"),
            _mkvar("airBoxLength", f"substrateLength + wavelength * {airbox_margin_wl}"),
            _mkvar("patchHeight", f"{patch_height_mm}mm"),
            _mkvar("GNDHeight", f"{gnd_height_mm}mm"),
            _mkvar("airBoxHeight",
                  f"substrateHeight + patchHeight + GNDHeight + wavelength * {airbox_margin_wl}"),
        ]]])

    # ── 基板 & 地 ──
    oEditor.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", "-substrateLength /2",
         "YPosition:=", "-substrateWidth / 2", "ZPosition:=", "0mm",
         "XSize:=", "substrateLength", "YSize:=", "substrateWidth",
         "ZSize:=", "substrateHeight"],
        _mkattr("Substrate", "(0 128 0)", "FR4_epoxy", solve_inside=True))
    oEditor.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", "-substrateLength / 2",
         "YPosition:=", "-substrateWidth / 2", "ZPosition:=", "- GNDHeight",
         "XSize:=", "substrateLength", "YSize:=", "substrateWidth",
         "ZSize:=", "GNDHeight"],
        _mkattr("GND", "(255 128 0)", "copper", solve_inside=False))

    # ── 阵元 ──
    print("AEP: Start modeling...")
    port_faces, port_holes, feed_centers = [], "", []
    for i in range(num_elems):
        idx = i + 1
        xc, yc = xc_mm[i], yc_mm[i]
        fx = f"{xc}mm + L1"

        oEditor.CreateBox(
            ["NAME:BoxParameters", "XPosition:=", f"{xc}mm - patchLength/2",
             "YPosition:=", f"{yc}mm - patchWidth / 2", "ZPosition:=", "substrateHeight",
             "XSize:=", "patchLength", "YSize:=", "patchWidth", "ZSize:=", "patchHeight"],
            _mkattr(f"Patch{idx}", "(255 128 0)", "copper"))
        oEditor.CreateCylinder(
            ["NAME:CylinderParameters", "XCenter:=", fx, "YCenter:=", f"{yc}mm",
             "ZCenter:=", "-GNDHeight", "Radius:=", "feedRadius",
             "Height:=", "GNDHeight + substrateHeight", "WhichAxis:=", "Z", "NumSides:=", "0"],
            _mkattr(f"Feed{idx}", "(255 128 0)", "copper"))
        oEditor.CreateCylinder(
            ["NAME:CylinderParameters", "XCenter:=", fx, "YCenter:=", f"{yc}mm",
             "ZCenter:=", "-GNDHeight", "Radius:=", "portRadius",
             "Height:=", "substrateHeight + GNDHeight", "WhichAxis:=", "Z", "NumSides:=", "0"],
            _mkattr(f"PortHole{idx}", "(143 175 143)", "vacuum", solve_inside=True))
        oEditor.CreateCircle(
            ["NAME:CircleParameters", "IsCovered:=", True,
             "XCenter:=", fx, "YCenter:=", f"{yc}mm", "ZCenter:=", "-GNDHeight",
             "Radius:=", "portRadius", "WhichAxis:=", "Z", "NumSegments:=", "0"],
            _mkattr(f"portFace{idx}", "(143 175 143)", "vacuum", solve_inside=True))

        port_faces.append(f"portFace{idx}")
        port_holes = f"PortHole{idx}" if i == 0 else port_holes + f", PortHole{idx}"
        feed_centers.append((xc + l1_mm, yc))

    # ── 布尔减 ──
    if port_holes:
        oEditor.Subtract(
            ["NAME:Selections", "Blank Parts:=", "Substrate, GND",
             "Tool Parts:=", port_holes],
            ["NAME:SubtractParameters", "KeepOriginals:=", False, "TurnOnNBodyBoolean:=", True])

    # ── 端口 (全部设为 LumpedPort, 50Ω) ──
    oModule = oDesign.GetModule("BoundarySetup")
    for idx, pf in enumerate(port_faces, start=1):
        fx, fy = feed_centers[idx - 1]
        s = [f"{fx + feed_radius_mm}mm", f"{fy}mm", f"-{gnd_height_mm}mm"]
        e = [f"{fx + port_radius_mm}mm", f"{fy}mm", f"-{gnd_height_mm}mm"]
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
        _mkattr("AirBox", "(143 175 143)", "air", solve_inside=True, transparency=0.8))
    oModule.AssignRadiation(["NAME:Rad1", "Objects:=", ["AirBox"]])

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

    print("AEP: Modeling completed.")

    # ══════════════════════════════════════════════════════
    # 2. 首次仿真 (自适应网格收敛)
    # ══════════════════════════════════════════════════════
    # 初始激励: 全部端口 1W/0° (仅用于网格收敛)
    oModuleSolutions = oDesign.GetModule("Solutions")
    _edit_sources(oModuleSolutions, num_elems, excite_port=None)  # None → 全部 1W
    print("AEP: Initial solve (all ports, mesh convergence)...")
    oDesign.Analyze("Setup1")
    print("AEP: Initial solve completed.")

    # ══════════════════════════════════════════════════════
    # 3. AEP 循环: 逐端口激励 → 重解 → 导出
    # ══════════════════════════════════════════════════════
    os.makedirs(results_dir, exist_ok=True)
    freq_str = f"{frequency_ghz:g}"
    csv_paths = []

    oModuleReport = oDesign.GetModule("ReportSetup")
    report_type = f"{results_function}({results_category})" if results_function else results_category
    phi_entries = "All" if not results_phi_sections else [f"{p}deg" for p in results_phi_sections]

    for elem_i in aep_elements:
        print(f"\nAEP: Simulating element {elem_i}/{num_elems} "
              f"(port {elem_i} = 1W, others = 0W)...")

        # 设置激励: 仅端口 elem_i = 1W/0°
        _edit_sources(oModuleSolutions, num_elems, excite_port=elem_i)

        # 重解 (网格已收敛, 速度快)
        oDesign.Analyze("Setup1")

        # 导出远场 CSV
        csv_name = f"aep_elem_{elem_i}_{freq_str}GHz.csv"
        export_path = os.path.normpath(os.path.join(results_dir, csv_name))

        report_name = f"AEP_Elem{elem_i}"
        oModuleReport.CreateReport(report_name, "Far Fields", "Rectangular Plot",
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
            oModuleReport.ExportUniformPointsToFile(report_name, export_path,
                f"{theta_start}deg", f"{theta_stop}deg",
                f"{theta_step}deg", True, "", False, True)
            if not os.path.isfile(export_path):
                csvs = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
                if csvs:
                    export_path = os.path.join(results_dir, csvs[-1])
                    print(f"  Exported to {export_path} (detected)")
                else:
                    print(f"  Export call succeeded but file not found: {export_path}")
            else:
                print(f"  Exported to {export_path}")
        except Exception as ex:
            print(f"  Export error: {ex}")

        csv_paths.append(export_path)

    print(f"\nAEP: All {len(aep_elements)} elements exported to {results_dir}")

    if close_after:
        oDesktop.CloseProject(oProject.GetName())
        oDesktop.QuitApplication()

    return csv_paths


# ════════════════════════════════════════════════════════════
#  内部辅助函数
# ════════════════════════════════════════════════════════════

def _edit_sources(oModuleSolutions, num_elems: int, excite_port: int = None):
    """设置端口激励。

    Args:
        excite_port: 要激励的端口号 (1-based), None=全部 1W
    """
    args = [["IncludePortPostProcessing:=", False, "SpecifySystemPower:=", False]]
    for i in range(1, num_elems + 1):
        if excite_port is None:
            # 全部激励 (首次仿真用)
            args.append(["Name:=", f"{i}:1", "Magnitude:=", "1W", "Phase:=", "0deg"])
        elif i == excite_port:
            # 仅激励目标端口
            args.append(["Name:=", f"{i}:1", "Magnitude:=", "1W", "Phase:=", "0deg"])
        else:
            # 其余端口 0W (等效匹配负载)
            args.append(["Name:=", f"{i}:1", "Magnitude:=", "0W", "Phase:=", "0deg"])
    oModuleSolutions.EditSources(args)


def _mkvar(name, value):
    return ["NAME:" + name, "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", value]


def _mkattr(name, color, material, solve_inside=False, transparency=0):
    return ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", color,
            "Transparency:=", transparency, "PartCoordinateSystem:=", "Global",
            "UDMId:=", "", "MaterialValue:=", f'"{material}"',
            "SurfaceMaterialValue:=", '""', "SolveInside:=", solve_inside,
            "ShellElement:=", False, "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel", "IsMaterialEditable:=", True,
            "IsSurfaceMaterialEditable:=", True, "UseMaterialAppearance:=", False,
            "IsLightweight:=", False]
