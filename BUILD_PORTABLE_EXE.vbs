Option Explicit

Dim fso, shell, rootPath, guiPath, mshtaPath, commandLine
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

rootPath = fso.GetParentFolderName(WScript.ScriptFullName)
guiPath = fso.BuildPath(rootPath, "installer\portable_build_gui.hta")
mshtaPath = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\mshta.exe"

If Not fso.FileExists(guiPath) Then
    MsgBox "Portable build GUI was not found:" & vbCrLf & guiPath, vbCritical, "Polyfitty Studio Builder"
    WScript.Quit 1
End If

If Not fso.FileExists(mshtaPath) Then
    mshtaPath = "mshta.exe"
End If

shell.CurrentDirectory = rootPath
commandLine = Chr(34) & mshtaPath & Chr(34) & " " & Chr(34) & guiPath & Chr(34)

On Error Resume Next
shell.Run commandLine, 1, False
If Err.Number <> 0 Then
    MsgBox "Unable to open the portable build window:" & vbCrLf & Err.Description, vbCritical, "Polyfitty Studio Builder"
    WScript.Quit 2
End If
On Error GoTo 0
