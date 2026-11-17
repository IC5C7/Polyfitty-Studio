Option Explicit

Dim fso, shell, sourceRoot, outputRoot, exePath, desktopPath
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

sourceRoot = fso.GetParentFolderName(WScript.ScriptFullName)
outputRoot = fso.BuildPath(sourceRoot, "release\portable")
exePath = fso.BuildPath(outputRoot, "PolyfittyStudio.exe")

If Not fso.FileExists(exePath) Then
    MsgBox "The single-file portable executable has not been built yet:" & vbCrLf & vbCrLf & _
        exePath & vbCrLf & vbCrLf & "Run BUILD_PORTABLE_EXE.bat first.", vbCritical, "Polyfitty Studio"
    WScript.Quit 1
End If

desktopPath = shell.SpecialFolders("Desktop")
CreatePortableShortcut fso.BuildPath(outputRoot, "Polyfitty Studio.lnk"), exePath, outputRoot
CreatePortableShortcut fso.BuildPath(desktopPath, "Polyfitty Studio.lnk"), exePath, outputRoot
MsgBox "Shortcuts to the single-file Polyfitty Studio executable were created.", vbInformation, "Polyfitty Studio"

Sub CreatePortableShortcut(shortcutPath, targetPath, workingDirectory)
    Dim shortcut
    Set shortcut = shell.CreateShortcut(shortcutPath)
    shortcut.TargetPath = targetPath
    shortcut.WorkingDirectory = workingDirectory
    shortcut.Description = "Polyfitty Studio - Single-file portable 3D mesh reconstruction workspace"
    shortcut.IconLocation = targetPath & ",0"
    shortcut.Save
End Sub
