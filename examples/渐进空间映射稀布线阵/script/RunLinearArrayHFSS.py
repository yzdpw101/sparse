import os
import math
import json

def runHFSS():
    hfssConfigFilePath = "E:\\Documents\\南理工\\阵列天线稀疏\\Sparse\\渐进空间映射稀布线阵\\script\\LinearArrayScriptConfig.json"
    hfssConfig = read_json_file(hfssConfigFilePath)
    # 保存阵元位置的 Json 文件路径
    elementConfigFilePath = hfssConfig["script"]["hfss"]["elementConfigFilePath"]

    # ---------------------------- 参数配置 ----------------------------
    # xCenters_mm = [0.0]
    # yCenters_mm = [0.0]
    # magnitude = [1.0]
    # phaseDeg = [0.0]

    # xCenters_mm = [0.0, 120.0, -150.0, 300.0, -250.0]
    # yCenters_mm = [0.0, 120.0, -180.0, -300.0, 250.0]
    # magnitude = []
    # phaseDeg = []

    # xCenters_mm = [0.0, -61.2244, 61.2244, 0, 0, -61.2244, 61.2244, -61.2244, 61.2244]
    # yCenters_mm = [0.0, 0, 0, 61.2244, -61.2244, 61.2244, 61.2244, -61.2244, -61.2244]
    # magnitude = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # phaseDeg = []

    elementConfig = read_json_file(elementConfigFilePath)
    xCenters = elementConfig["xCenters"]
    yCenters = elementConfig["yCenters"]
    magnitude = elementConfig.get("magnitudes", [])
    phaseDeg = elementConfig.get("phasesDeg", [])
    xCenters_mm = [x * 1000 for x in xCenters]
    yCenters_mm = [y * 1000 for y in yCenters]

    # print(xCenters_mm)

    xCenters_mm = [0.0]
    yCenters_mm = [0.0]
    magnitude = [1.0]
    phaseDeg = [0.0]

    ##########################################################################
    # 工程设置
    pattern_export_directory = hfssConfig["script"]["hfss"]["pattern_export_directory"]
    project_save_path = hfssConfig["script"]["hfss"]["project_save_path"]
    close_after_simulation = hfssConfig["script"]["hfss"]["close_after_simulation"]
    run_simulation = hfssConfig["script"]["hfss"]["run_simulation"]

    # 导出的远场方向图配置
    Results_Category = hfssConfig["script"]["hfss"]["Results_Category"]
    Results_Function = hfssConfig["script"]["hfss"]["Results_Function"]
    Results_Sections_deg = hfssConfig["script"]["hfss"]["Results_Sections_deg"]
    Results_ThetaStart_deg = hfssConfig["script"]["hfss"]["Results_ThetaStart_deg"]
    Results_ThetaStop_deg = hfssConfig["script"]["hfss"]["Results_ThetaStop_deg"]
    Results_ThetaStep_deg = hfssConfig["script"]["hfss"]["Results_ThetaStep_deg"]

    # 变量设置
    lightSpeed = 299792458
    frequency_GHz = hfssConfig["script"]["hfss"]["frequency_GHz"]
    epsilon_r = hfssConfig["script"]["hfss"]["epsilon_r"]
    patchLength_mm = hfssConfig["script"]["hfss"]["patchLength_mm"]
    patchHeight_mm = hfssConfig["script"]["hfss"]["patchHeight_mm"]
    arrayWidth_wavelength = hfssConfig["script"]["hfss"]["arrayWidth_wavelength"]
    arrayLength_wavelength = hfssConfig["script"]["hfss"]["arrayLength_wavelength"]
    substrateHeight_mm = hfssConfig["script"]["hfss"]["substrateHeight_mm"]
    arrayMargin_wavelength = hfssConfig["script"]["hfss"]["arrayMargin_wavelength"]
    airboxMargin_wavelength = hfssConfig["script"]["hfss"]["airboxMargin_wavelength"]
    GNDHeight_mm = hfssConfig["script"]["hfss"]["GNDHeight_mm"]
    portRadius_mm = hfssConfig["script"]["hfss"]["portRadius_mm"]
    feedRadius_mm = hfssConfig["script"]["hfss"]["feedRadius_mm"]
    L1_mm = hfssConfig["script"]["hfss"]["L1_mm"]

    # 单频求解器配置
    InsertSetup_MaxDeltaS = hfssConfig["script"]["hfss"]["InsertSetup_MaxDeltaS"]
    InsertSetup_MaximumPasses = hfssConfig["script"]["hfss"]["InsertSetup_MaximumPasses"]
    InsertSetup_SaveRadFieldsOnly = hfssConfig["script"]["hfss"]["InsertSetup_SaveRadFieldsOnly"]
    
    # 远场辐射边界配置
    ThetaStart_deg = hfssConfig["script"]["hfss"]["ThetaStart_deg"]
    ThetaStop_deg = hfssConfig["script"]["hfss"]["ThetaStop_deg"]
    ThetaStep_deg = hfssConfig["script"]["hfss"]["ThetaStep_deg"]
    PhiStart_deg = hfssConfig["script"]["hfss"]["PhiStart_deg"]
    PhiStop_deg = hfssConfig["script"]["hfss"]["PhiStop_deg"]
    PhiStep_deg = hfssConfig["script"]["hfss"]["PhiStep_deg"]

    # 激励类型
    Assign_Excitation_Type = hfssConfig["script"]["hfss"]["Assign_Excitation_Type"]

    # -----------------------------------------------------------------

    import win32com.client
    # 初始化HFSS接口
    oAnsoftApp = win32com.client.Dispatch("Ansoft.ElectronicsDesktop")
    oDesktop = oAnsoftApp.GetAppDesktop()
    oDesktop.RestoreWindow()

    oProject = oDesktop.NewProject()
    oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Modal Network", "")

    oDesign = oProject.SetActiveDesign("HFSSDesign1")
    oEditor = oDesign.SetActiveEditor("3D Modeler")

    # 定义变量
    oDesign.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:LocalVariableTab",
                [
                    "NAME:PropServers", 
                    "LocalVariables"
                ],
                [
                    "NAME:NewProps",
                    [
                        "NAME:patchLength",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{patchLength_mm}mm"
                    ],
                    [
                        "NAME:frequency",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{frequency_GHz}e9"
                    ],
                    [
                        "NAME:lightSpeed",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{lightSpeed}"
                    ],
                    [
                        "NAME:wavelength",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{lightSpeed / frequency_GHz / 1e9 * 1000}mm"
                    ],
                    [
                        "NAME:epsilon_r",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{epsilon_r}"
                    ],
                    [
                        "NAME:portRadius",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{portRadius_mm}mm"
                    ],
                    [
                        "NAME:feedRadius",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{feedRadius_mm}mm"
                    ],
                    [
                        "NAME:L1",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{L1_mm}mm"
                    ],
                    [
                        "NAME:patchWidth",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{lightSpeed / (2 * frequency_GHz * 1e9) * math.sqrt(2 / (epsilon_r + 1)) * 1000}mm"
                    ],
                    [
                        "NAME:substrateWidth",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"wavelength * {arrayWidth_wavelength} + wavelength * {arrayMargin_wavelength}"
                    ],
                    [
                        "NAME:substrateLength",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"wavelength * {arrayLength_wavelength} + wavelength * {arrayMargin_wavelength}"
                    ],
                    [
                        "NAME:substrateHeight",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{substrateHeight_mm}mm"
                    ],
                    [
                        "NAME:airBoxWidth",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"substrateWidth + wavelength * {airboxMargin_wavelength}"
                    ],
                    [
                        "NAME:airBoxLength",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"substrateLength + wavelength * {airboxMargin_wavelength}"
                    ],
                    [
                        "NAME:patchHeight",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{patchHeight_mm}mm"
                    ],
                    [
                        "NAME:GNDHeight",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{GNDHeight_mm}mm"
                    ],
                    [
                        "NAME:airBoxHeight",
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"substrateHeight + patchHeight + GNDHeight + wavelength * {airboxMargin_wavelength}"
                    ]
                ]
            ]
        ])

    # 创建基板
    oEditor.CreateBox(
        [
            "NAME:BoxParameters",
            "XPosition:="		, "-substrateLength /2",
            "YPosition:="		, "-substrateWidth / 2",
            "ZPosition:="		, "0mm",
            "XSize:="		, "substrateLength",
            "YSize:="		, "substrateWidth",
            "ZSize:="		, "substrateHeight"
        ], 
        [
            "NAME:Attributes",
            "Name:="		, "Substrate",
            "Flags:="		, "",
            "Color:="		, "(0 128 0)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"FR4_epoxy\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])

    # 创建地平面
    oEditor.CreateBox(
        [
            "NAME:BoxParameters",
            "XPosition:="		, "-substrateLength / 2",
            "YPosition:="		, "-substrateWidth / 2",
            "ZPosition:="		, "- GNDHeight",
            "XSize:="		, "substrateLength",
            "YSize:="		, "substrateWidth",
            "ZSize:="		, "GNDHeight"
        ], 
        [
            "NAME:Attributes",
            "Name:="		, "GND",
            "Flags:="		, "",
            "Color:="		, "(255 128 0)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"copper\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, False,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])


    # ------------------------------------ 设置天线单元 -------------------------------------
    print("Start modeling...")
    numElems = min(len(xCenters_mm), len(yCenters_mm))

    portFaceNames = []      # 用于后续创建端口
    portHoleNames = ""      # 用于最后一次性布尔运算
    feedCenters = []        # 存储每个馈点的物理坐标（mm），用于端口 IntLine

    for i in range(numElems):
        idx = i + 1
        xc = xCenters_mm[i] # 注意馈点相对于贴片中心在 X 方向偏移 L1
        yc = yCenters_mm[i]

        # 1) 创建 Patch（贴片）
        oEditor.CreateBox(
            [
                "NAME:BoxParameters",
                "XPosition:="    , f"{xc}mm - patchLength/2",
                "YPosition:="    , f"{yc}mm - patchWidth / 2",
                "ZPosition:="    , "substrateHeight",
                "XSize:="        , "patchLength",
                "YSize:="        , "patchWidth",
                "ZSize:="        , "patchHeight"
            ],
            [
                "NAME:Attributes",
                "Name:="         , f"Patch{idx}",
                "Flags:="        , "",
                "Color:="        , "(255 128 0)",
                "Transparency:=" , 0,
                "PartCoordinateSystem:=", "Global",
                "MaterialValue:=" , "\"copper\"",
                "SolveInside:="  , False
            ])

        # 2) 创建 Feed（圆柱），相对于 Patch 中心在 X 方向偏移 L1
        feed_x_expr = f"{xc}mm + L1"
        oEditor.CreateCylinder(
            [
                "NAME:CylinderParameters",
                "XCenter:="    , feed_x_expr,
                "YCenter:="    , f"{yc}mm",
                "ZCenter:="    , "-GNDHeight",
                "Radius:="     , "feedRadius",
                "Height:="     , "GNDHeight + substrateHeight",
                "WhichAxis:="  , "Z",
                "NumSides:="   , "0"
            ],
            [
                "NAME:Attributes",
                "Name:="       , f"Feed{idx}",
                "Flags:="      , "",
                "Color:="      , "(255 128 0)",
                "Transparency:=", 0,
                "PartCoordinateSystem:=", "Global",
                "MaterialValue:=", "\"copper\"",
                "SolveInside:=", False
            ])

        # 3) 创建 PortHole（在基板上开的孔，用于打通）
        oEditor.CreateCylinder(
            [
                "NAME:CylinderParameters",
                "XCenter:="    , feed_x_expr,
                "YCenter:="    , f"{yc}mm",
                "ZCenter:="    , "-GNDHeight",
                "Radius:="     , "portRadius",
                "Height:="     , "substrateHeight + GNDHeight",
                "WhichAxis:="  , "Z",
                "NumSides:="   , "0"
            ],
            [
                "NAME:Attributes",
                "Name:="       , f"PortHole{idx}",
                "Flags:="      , "",
                "Color:="      , "(143 175 143)",
                "Transparency:=", 0,
                "PartCoordinateSystem:=", "Global",
                "MaterialValue:=", "\"vacuum\"",
                "SolveInside:=", True
            ])

        # 4) 在孔位置创建 portFace（圆面）, 作为端口对象
        oEditor.CreateCircle(
            [
                "NAME:CircleParameters",
                "IsCovered:="  , True,
                "XCenter:="    , feed_x_expr,
                "YCenter:="    , f"{yc}mm",
                "ZCenter:="    , "-GNDHeight",
                "Radius:="     , "portRadius",
                "WhichAxis:="  , "Z",
                "NumSegments:=","0"
            ],
            [
                "NAME:Attributes",
                "Name:="       , f"portFace{idx}",
                "Flags:="      , "",
                "Color:="      , "(143 175 143)",
                "Transparency:=", 0,
                "PartCoordinateSystem:=", "Global",
                "MaterialValue:=", "\"vacuum\"",
                "SolveInside:=", True
            ])

        # 记录名称与馈点实际坐标（mm），方便后面分配端口时设置 IntLine
        portFaceNames.append(f"portFace{idx}")
        # 拼接：如果是第一个，不加前导逗号；否则加 ", "
        if i == 0:
            portHoleNames = f"PortHole{idx}"
        else:
            portHoleNames += ", " + f"PortHole{idx}"
        # feed center 的数值坐标（单位 mm）用于 AssignLumpedPort 中的 IntLine
        feedCenters.append((xc + L1_mm, yc))

    # 结束循环

    # 5) 最后一次性对 Substrate 和 GND 做布尔减（把所有的 PortHole 从 Substrate/GND 中减掉）
    if portHoleNames:
        # 从 Substrate 和 GND 中减去所有孔（如果需要保留原件，请把 KeepOriginals 改为 True）
        oEditor.Subtract(
            [
                "NAME:Selections",
                "Blank Parts:="  , "Substrate, GND",
                "Tool Parts:="   , portHoleNames
            ],
            [
                "NAME:SubtractParameters",
                "KeepOriginals:="    , False,
                "TurnOnNBodyBoolean:=", True
            ])

    # 6) 根据刚创建的 portFaceNames 循环设置集总端口（Lumped Port）
    
    oModule = oDesign.GetModule("BoundarySetup")
    if Assign_Excitation_Type == "Voltage":
        for idx, pfName in enumerate(portFaceNames, start=1):
            fx_mm, fy_mm = feedCenters[idx-1]   # 对应的真实坐标（float, mm）
            # IntLine 起止取在 feed 半径与 port 半径之间（与原脚本一致的方向）
            start = [f"{fx_mm + feedRadius_mm}mm", f"{fy_mm}mm", f"-{GNDHeight_mm}mm"]
            end   = [f"{fx_mm + portRadius_mm}mm", f"{fy_mm}mm", f"-{GNDHeight_mm}mm"]
            oModule.AssignVoltage(
	            [
		            "NAME:" + str(idx),
		            "Objects:="		, [pfName],
		            [
			            "NAME:Direction",
			            "Coordinate System:="	, "Global",
			            "Start:=" , start,
                        "End:="   , end
		            ]
	            ])
    elif Assign_Excitation_Type == "LumpedPort":
        for idx, pfName in enumerate(portFaceNames, start=1):
            fx_mm, fy_mm = feedCenters[idx-1]   # 对应的真实坐标（float, mm）
            # IntLine 起止取在 feed 半径与 port 半径之间（与原脚本一致的方向）
            start = [f"{fx_mm + feedRadius_mm}mm", f"{fy_mm}mm", f"-{GNDHeight_mm}mm"]
            end   = [f"{fx_mm + portRadius_mm}mm", f"{fy_mm}mm", f"-{GNDHeight_mm}mm"]
            oModule.AssignLumpedPort(
                [
                    "NAME:" + str(idx),
                    "Objects:="    , [pfName],
                    "LumpedPortType:=" , "Default",
                    "DoDeembed:="  , False,
                    [
                        "NAME:Modes",
                        [
                            "NAME:Mode1",
                            "ModeNum:=" , 1,
                            "UseIntLine:=", True,
                            [
                                "NAME:IntLine",
                                "Coordinate System:=" , "Global",
                                "Start:=" , start,
                                "End:="   , end
                            ],
                            "AlignmentGroup:=" , 0,
                            "CharImp:=" , "Zpi",
                            "RenormImp:=" , "50ohm"
                        ]
                    ],
                    "Impedance:=" , "50ohm"
                ])

    # 创建空气盒子
    oEditor.CreateBox(
        [
            "NAME:BoxParameters",
            "XPosition:="		, "-airBoxLength / 2",
            "YPosition:="		, "-airBoxWidth / 2",
            # "ZPosition:="		, "-10mm",
            "ZPosition:="		, "-airBoxHeight / 2",
            "XSize:="		, "airBoxLength",
            "YSize:="		, "airBoxWidth",
            "ZSize:="		, "airBoxHeight"
        ], 
        [
            "NAME:Attributes",
            "Name:="		, "AirBox",
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0.8,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"air\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "ReferenceTemperature:=", "20cel",
            "IsMaterialEditable:="	, True,
            "IsSurfaceMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])

    # 设置辐射边界
    oModule.AssignRadiation(
        [
            "NAME:Rad1",
            "Objects:="		, ["AirBox"]
        ])

    # 设置源
    oModuleSolutions = oDesign.GetModule("Solutions")
    if magnitude != []:
        mag = magnitude
        print(f"Loading amplitude.")
    else:
        mag = [1.0] * numElems
    if phaseDeg != []:
        phase = phaseDeg
        print(f"Loading phase.")
    else:
        phase = [0.0] * numElems
    EditSources_args = [
        [  # 第一组：全局设置
            "IncludePortPostProcessing:=", False,
            "SpecifySystemPower:=", False
        ]
        ] + [
            [  # 每个激励一项
                "Name:=",        f"{i+1}:1",
                "Magnitude:=",   f"{mag[i]}W",
                "Phase:=",       f"{phase[i]}deg"
            ]
            for i in range(numElems)
        ]
    oModuleSolutions.EditSources(EditSources_args)

    # 设置远场辐射边界
    InsertInfiniteSphereSetup_Name = "Infinite Sphere1"
    oModule = oDesign.GetModule("RadField")
    oModule.InsertInfiniteSphereSetup(
        [
            f"NAME:{InsertInfiniteSphereSetup_Name}",
            "UseCustomRadiationSurface:=", False,
            "CSDefinition:="	, "Theta-Phi",
            "Polarization:="	, "Linear",
            "ThetaStart:="		, f"{ThetaStart_deg}deg",
            "ThetaStop:="		, f"{ThetaStop_deg}deg",
            "ThetaStep:="		, f"{ThetaStep_deg}deg",
            "PhiStart:="		, f"{PhiStart_deg}deg",
            "PhiStop:="		, f"{PhiStop_deg}deg",
            "PhiStep:="		, f"{PhiStep_deg}deg",
            "UseLocalCS:="		, False
        ])

    # 设置求解
    oModule = oDesign.GetModule("AnalysisSetup")
    oModule.InsertSetup("HfssDriven", 
        [
            "NAME:Setup1",
            "SolveType:="		, "Single",
            "Frequency:="		, f"{frequency_GHz}GHz",
            "MaxDeltaS:="		, InsertSetup_MaxDeltaS,
            "UseMatrixConv:="	, False,
            "MaximumPasses:="	, InsertSetup_MaximumPasses,
            "MinimumPasses:="	, 1,
            "MinimumConvergedPasses:=", 1,
            "PercentRefinement:="	, 30,
            "IsEnabled:="		, True,
            [
                "NAME:MeshLink",
                "ImportMesh:="		, False
            ],
            "BasisOrder:="		, 1,
            "DoLambdaRefine:="	, True,
            "DoMaterialLambda:="	, True,
            "SetLambdaTarget:="	, False,
            "Target:="		, 0.3333,
            "UseMaxTetIncrease:="	, False,
            "PortAccuracy:="	, 2,
            "UseABCOnPort:="	, False,
            "SetPortMinMaxTri:="	, False,
            "DrivenSolverType:="	, "Direct Solver",
            "EnhancedLowFreqAccuracy:=", False,
            "SaveRadFieldsOnly:="	, InsertSetup_SaveRadFieldsOnly,
            "SaveAnyFields:="	, True,
            "IESolverType:="	, "Auto",
            "LambdaTargetForIESolver:=", 0.15,
            "UseDefaultLambdaTgtForIESolver:=", True,
            "IE Solver Accuracy:="	, "Balanced",
            "InfiniteSphereSetup:="	, InsertInfiniteSphereSetup_Name,
            "MaxPass:="		, 10,
            "MinPass:="		, 1,
            "MinConvPass:="		, 1,
            "PerError:="		, 1,
            "PerRefine:="		, 30
        ])

    # 保存项目
    if project_save_path != "":
        oProject.SaveAs(project_save_path, True)
        print(f"Project saved to: {project_save_path}")
    print("Modeling completed")

    # ---------------------------- 仿真与后处理 ----------------------------
    if run_simulation:
        # 执行仿真
        print("Starting simulation...")
        oDesign.Analyze("Setup1")
        print("Simulation completed.")

        # 获取报告模块
        oModule = oDesign.GetModule("ReportSetup")

        # 确保导出目录存在
        if not os.path.isdir(pattern_export_directory):
            os.makedirs(pattern_export_directory, exist_ok=True)

        reportType = f"{Results_Function}({Results_Category})"

        # 转换为字符列表，加上deg
        if Results_Sections_deg == []:
            phi_degs = "All"
        else:
            phi_degs = [f"{phi}deg" for phi in Results_Sections_deg]
        # report_name = f"Realized Gain Plot"
        report_name = Results_Category
        # 导出文件名
        export_filepath = f"{pattern_export_directory}{reportType}.csv"

        # 创建远场方向图报告
        oModule.CreateReport(report_name, "Far Fields", "Rectangular Plot", "Setup1 : LastAdaptive", 
            [
                "Context:="		, "Infinite Sphere1"
            ], 
            [
                "Theta:="		, ["All"],
                "Phi:="			, [phi_degs], # 使用当前循环的 phi 角度
                "Freq:="		, [f"{frequency_GHz}GHz"], # 使用配置的频率
                # 包含所有变量的名义值
                "patchLength:="		, ["Nominal"],
                "frequency:="	    , ["Nominal"],
                "lightSpeed:="		, ["Nominal"],
                "epsilon_r:="		, ["Nominal"],
                "portRadius:="		, ["Nominal"],
                "feedRadius:="		, ["Nominal"],
                "L1:="			    , ["Nominal"],
                "substrateHeight:="	, ["Nominal"],
                "patchHeight:="		, ["Nominal"],
                "GNDHeight:="		, ["Nominal"]
            ], 
            [
                "X Component:="		, "Theta",
                "Y Component:="		, [reportType] # 导出 Realized Gain
            ])

        # 导出数据到 CSV 文件
        # 注意：确保 pattern_export_directory 指定的文件夹存在
        try:
            oModule.ExportUniformPointsToFile(report_name, export_filepath, 
                                                f"{Results_ThetaStart_deg}deg", f"{Results_ThetaStop_deg}deg", f"{Results_ThetaStep_deg}deg", 
                                                True, "", False, True)
            print(f"Exported {report_name} to {export_filepath}")
        except Exception as e:
            print(f"Error exporting {report_name}: {e}")

        print("Simulation and export completed.")
    else:
        print("Simulation and export skipped based on 'run_simulation' setting.")

    print("Script execution completed.")

    # 是否关闭 HFSS 软件
    if close_after_simulation:
        # 1) 关闭当前工程（使用 Desktop API，传入工程名）
        projectName = oProject.GetName()  # 返回工程名（不带路径）
        oDesktop.CloseProject(projectName)  # 也可用 oProject.Close()
        # 2) 退出 Electronics Desktop（关闭 HFSS 窗口并结束应用）
        oDesktop.QuitApplication()
        print("HFSS software closed.")
    # -----------------------------------------------------------------

# 读取 JSON 文件
def read_json_file(file_path):
    """
    读取 JSON 文件并返回其内容
    
    参数:
        file_path (str): JSON 文件的路径
        
    返回:
        dict: JSON 文件的内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件不是有效的 JSON 格式 - {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

if __name__ == '__main__':
    runHFSS()  # 执行 main，此时 greet 已定义