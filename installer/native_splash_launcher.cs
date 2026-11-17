using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[assembly: AssemblyTitle("Polyfitty Studio")]
[assembly: AssemblyDescription("Single-file portable 3D mesh reconstruction and baking studio")]
[assembly: AssemblyCompany("Polyfitty Studio")]
[assembly: AssemblyProduct("Polyfitty Studio")]
[assembly: AssemblyVersion("0.2.131.0")]
[assembly: AssemblyFileVersion("0.2.131.0")]
[assembly: AssemblyInformationalVersion("0.2.131")]

internal static class PolyfittyStudioLauncher
{
    [STAThread]
    private static int Main(string[] args)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);

        string powerShellPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.Windows),
            "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
        if (!File.Exists(powerShellPath))
        {
            MessageBox.Show(
                "Windows PowerShell 5.1 is required to initialize Polyfitty Studio.",
                "Polyfitty Studio",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 2;
        }

        string token = Guid.NewGuid().ToString("N");
        string readyEventName = "Local\\PolyfittyStudio_Ready_" + token;
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string statusRoot = Path.Combine(localAppData, "PolyfittyStudio", "Portable", "logs");
        string statusPath = Path.Combine(statusRoot, "startup_" + token + ".txt");

        try
        {
            Directory.CreateDirectory(statusRoot);
            using (EventWaitHandle readyEvent = new EventWaitHandle(false, EventResetMode.ManualReset, readyEventName))
            using (SplashForm splash = new SplashForm(powerShellPath, statusPath, readyEventName, readyEvent))
            {
                Application.Run(splash);
                return splash.ExitCode;
            }
        }
        catch (Exception ex)
        {
            MessageBox.Show(
                "Polyfitty Studio could not start.\r\n\r\n" + ex.Message,
                "Polyfitty Studio",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
            return 3;
        }
        finally
        {
            try { if (File.Exists(statusPath)) File.Delete(statusPath); } catch { }
        }
    }
}

internal static class EmbeddedPortablePayload
{
    private const string PayloadResourceName = "PolyfittyStudio.Payload.zip";
    private const string PayloadHashResourceName = "PolyfittyStudio.Payload.sha256";
    private const string CompletionMarkerName = ".portable_payload_complete";

    public static string Extract(Action<string> report)
    {
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (String.IsNullOrWhiteSpace(localAppData))
            throw new InvalidOperationException("Windows did not provide a Local AppData folder for the embedded portable runtime.");

        string productRoot = Path.Combine(localAppData, "PF");
        string packageRootDirectory = Path.Combine(productRoot, "P");
        string stagingRootDirectory = Path.Combine(productRoot, "T");
        Directory.CreateDirectory(packageRootDirectory);
        Directory.CreateDirectory(stagingRootDirectory);
        string extractionRoot = null;

        using (Stream resource = Assembly.GetExecutingAssembly().GetManifestResourceStream(PayloadResourceName))
        {
            if (resource == null)
                throw new InvalidOperationException("The single-file EXE does not contain its embedded application payload.");
            if (!resource.CanSeek)
                throw new InvalidOperationException("The embedded application payload stream does not support safe verification and extraction.");

            string payloadHash = ReadEmbeddedPayloadHash();
            string packageRoot = Path.Combine(packageRootDirectory, payloadHash);
            string mutexName = "Local\\PolyfittyStudio_PortablePayload_" + payloadHash;
            report("Checking the embedded application package...");

            using (Mutex packageMutex = new Mutex(false, mutexName))
            {
                bool ownsMutex = false;
                try
                {
                    try { ownsMutex = packageMutex.WaitOne(TimeSpan.FromMinutes(5)); }
                    catch (AbandonedMutexException) { ownsMutex = true; }
                    if (!ownsMutex)
                        throw new TimeoutException("Another Polyfitty Studio process is still preparing this embedded package.");

                    if (IsCompletePackage(packageRoot, payloadHash))
                    {
                        report("Embedded application is already prepared. Checking the recipient PC GPU...");
                        return packageRoot;
                    }

                    if (Directory.Exists(packageRoot))
                        Directory.Delete(packageRoot, true);

                    foreach (string stale in Directory.GetDirectories(stagingRootDirectory))
                    {
                        try
                        {
                            DirectoryInfo info = new DirectoryInfo(stale);
                            if (info.CreationTimeUtc < DateTime.UtcNow.AddDays(-1))
                                info.Delete(true);
                        }
                        catch { }
                    }

                    string actualPayloadHash = ComputeStreamSha256(resource);
                    if (!String.Equals(actualPayloadHash, payloadHash, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException("The embedded application payload failed its SHA-256 integrity check.");

                    extractionRoot = Path.Combine(stagingRootDirectory, Guid.NewGuid().ToString("N"));
                    Directory.CreateDirectory(extractionRoot);
                    report("Unpacking the embedded Python, Qt, VTK and application components...");
                    resource.Position = 0;
                    using (ZipArchive archive = new ZipArchive(resource, ZipArchiveMode.Read, true))
                        ExtractZipSafely(archive, extractionRoot, report);
                    ValidateExtractedPackage(extractionRoot);

                    File.WriteAllText(
                        Path.Combine(extractionRoot, CompletionMarkerName),
                        "Polyfitty Studio embedded portable payload\r\n" + payloadHash + "\r\n",
                        Encoding.UTF8);

                    Directory.Move(extractionRoot, packageRoot);
                    extractionRoot = null;
                    report("Embedded application prepared. Detecting this PC's GPU...");
                    return packageRoot;
                }
                finally
                {
                    if (!String.IsNullOrEmpty(extractionRoot) && Directory.Exists(extractionRoot))
                    {
                        try { Directory.Delete(extractionRoot, true); } catch { }
                    }
                    if (ownsMutex)
                    {
                        try { packageMutex.ReleaseMutex(); } catch { }
                    }
                }
            }
        }
    }

    private static string ReadEmbeddedPayloadHash()
    {
        using (Stream resource = Assembly.GetExecutingAssembly().GetManifestResourceStream(PayloadHashResourceName))
        {
            if (resource == null)
                throw new InvalidOperationException("The single-file EXE does not contain its embedded payload integrity record.");
            using (StreamReader reader = new StreamReader(resource, Encoding.ASCII))
            {
                string hash = reader.ReadToEnd().Trim();
                if (hash.Length != 64)
                    throw new InvalidDataException("The embedded payload integrity record is malformed.");
                for (int i = 0; i < hash.Length; i++)
                {
                    char c = hash[i];
                    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                        throw new InvalidDataException("The embedded payload integrity record is malformed.");
                }
                return hash.ToLowerInvariant();
            }
        }
    }

    private static string ComputeStreamSha256(Stream stream)
    {
        if (stream == null || !stream.CanSeek)
            throw new InvalidOperationException("The embedded application payload cannot be rewound for verification.");
        stream.Position = 0;
        using (SHA256 sha = SHA256.Create())
        {
            string hash = ToHex(sha.ComputeHash(stream));
            stream.Position = 0;
            return hash;
        }
    }

    private static string ToHex(byte[] bytes)
    {
        StringBuilder result = new StringBuilder(bytes.Length * 2);
        for (int i = 0; i < bytes.Length; i++)
            result.Append(bytes[i].ToString("x2"));
        return result.ToString();
    }

    private static bool IsCompletePackage(string root, string expectedHash)
    {
        try
        {
            string marker = Path.Combine(root, CompletionMarkerName);
            if (!File.Exists(marker)) return false;
            string markerText = File.ReadAllText(marker, Encoding.UTF8);
            if (markerText.IndexOf(expectedHash, StringComparison.OrdinalIgnoreCase) < 0) return false;
            ValidateExtractedPackage(root);
            return true;
        }
        catch { return false; }
    }

    private static void ValidateExtractedPackage(string root)
    {
        string[] requiredFiles = new string[] {
            Path.Combine(root, ".runtime", "python312", "python.exe"),
            Path.Combine(root, "installer", "start_app.ps1"),
            Path.Combine(root, "installer", "gpu_setup_state.ps1"),
            Path.Combine(root, "app", "hidden_entry.py"),
            Path.Combine(root, "portable_runtime_manifest.json"),
            Path.Combine(root, "runtime_manifest.json"),
            Path.Combine(root, "requirements.txt")
        };
        for (int i = 0; i < requiredFiles.Length; i++)
        {
            if (!File.Exists(requiredFiles[i]))
                throw new InvalidDataException("The embedded Polyfitty Studio payload is incomplete: " + requiredFiles[i]);
        }
    }

    private static void ExtractZipSafely(ZipArchive archive, string destinationRoot, Action<string> report)
    {
        string root = Path.GetFullPath(destinationRoot).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string rootPrefix = root + Path.DirectorySeparatorChar;
        long copiedBytes = 0;
        int extractedFiles = 0;
        long nextReportBytes = 48L * 1024L * 1024L;

        for (int i = 0; i < archive.Entries.Count; i++)
        {
            ZipArchiveEntry entry = archive.Entries[i];
            string relative = (entry.FullName ?? String.Empty).Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
            if (String.IsNullOrWhiteSpace(relative) || Path.IsPathRooted(relative) || relative.IndexOf(':') >= 0)
                throw new InvalidDataException("The embedded payload contains an invalid path.");

            int unixMode = (entry.ExternalAttributes >> 16) & 0xF000;
            if (unixMode == 0xA000)
                throw new InvalidDataException("The embedded payload contains an unsupported symbolic link.");

            bool isDirectory = entry.FullName.EndsWith("/", StringComparison.Ordinal) || entry.FullName.EndsWith("\\", StringComparison.Ordinal);
            string combined = Path.Combine(root, relative);
            string combinedParent = Path.GetDirectoryName(combined);

            if (isDirectory && combined.Length >= 248)
                throw new PathTooLongException("The embedded package contains a directory path that exceeds the Windows 248-character directory limit: " + relative);
            if (!isDirectory && (combinedParent != null && combinedParent.Length >= 248 || combined.Length >= 260))
                throw new PathTooLongException("The embedded package contains a file path that exceeds the Windows MAX_PATH limit: " + relative);

            string target = Path.GetFullPath(combined);
            if (!target.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("The embedded payload attempted to write outside its extraction directory.");
            string parent = Path.GetDirectoryName(target);
            if (isDirectory)
            {
                Directory.CreateDirectory(target);
                continue;
            }

            if (!Directory.Exists(parent)) Directory.CreateDirectory(parent);
            using (Stream source = entry.Open())
            using (FileStream output = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                byte[] buffer = new byte[1024 * 1024];
                int read;
                while ((read = source.Read(buffer, 0, buffer.Length)) > 0)
                {
                    output.Write(buffer, 0, read);
                    copiedBytes += read;
                }
            }

            extractedFiles++;
            if (copiedBytes >= nextReportBytes)
            {
                report("Unpacking embedded Studio components... " + extractedFiles.ToString() + " files");
                nextReportBytes = copiedBytes + 48L * 1024L * 1024L;
            }
        }
    }
}

internal sealed class SplashForm : Form
{
    private string _root;
    private string _runtimePath;
    private string _startupScript;
    private readonly string _powerShellPath;
    private readonly string _statusPath;
    private readonly string _readyEventName;
    private readonly EventWaitHandle _readyEvent;
    private readonly System.Windows.Forms.Timer _timer;
    private readonly MarqueeBar _progress;
    private readonly Label _status;
    private readonly Button _closeButton;
    private readonly object _payloadSync = new object();
    private Process _process;
    private bool _payloadStarted;
    private bool _payloadFinished;
    private bool _runtimeStarted;
    private bool _failureShown;
    private bool _allowClose;
    private string _payloadRoot;
    private string _payloadError;
    private string _payloadStatus;
    private int _phase;

    public int ExitCode { get; private set; }

    public SplashForm(string powerShellPath, string statusPath, string readyEventName, EventWaitHandle readyEvent)
    {
        _powerShellPath = powerShellPath;
        _statusPath = statusPath;
        _readyEventName = readyEventName;
        _readyEvent = readyEvent;
        _payloadStatus = "Preparing the single-file Polyfitty Studio application...";
        ExitCode = 0;

        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.CenterScreen;
        ShowInTaskbar = false;
        TopMost = true;
        ClientSize = new Size(760, 420);
        BackColor = Color.FromArgb(5, 15, 35);
        DoubleBuffered = true;
        BackgroundImageLayout = ImageLayout.Stretch;
        LoadSplashImage();

        _progress = new MarqueeBar();
        _progress.Location = new Point(282, 305);
        _progress.Size = new Size(423, 9);
        Controls.Add(_progress);

        _status = new Label();
        _status.AutoSize = false;
        _status.Location = new Point(282, 320);
        _status.Size = new Size(423, 24);
        _status.BackColor = Color.Transparent;
        _status.ForeColor = Color.FromArgb(183, 223, 255);
        _status.Font = new Font("Segoe UI", 9.0F, FontStyle.Regular, GraphicsUnit.Point);
        _status.TextAlign = ContentAlignment.MiddleLeft;
        _status.Text = _payloadStatus;
        Controls.Add(_status);

        _closeButton = new Button();
        _closeButton.Text = "Close";
        _closeButton.Location = new Point(648, 365);
        _closeButton.Size = new Size(74, 28);
        _closeButton.Visible = false;
        _closeButton.FlatStyle = FlatStyle.Flat;
        _closeButton.BackColor = Color.FromArgb(25, 59, 91);
        _closeButton.ForeColor = Color.White;
        _closeButton.FlatAppearance.BorderColor = Color.FromArgb(49, 153, 220);
        _closeButton.Click += delegate { _allowClose = true; Close(); };
        Controls.Add(_closeButton);

        FormClosing += delegate(object sender, FormClosingEventArgs e)
        {
            if (!_allowClose && !_failureShown)
                e.Cancel = true;
        };

        _timer = new System.Windows.Forms.Timer();
        _timer.Interval = 120;
        _timer.Tick += OnTimerTick;
        Shown += delegate { _timer.Start(); };
    }

    private void LoadSplashImage()
    {
        using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("PolyfittyStudio.Splash.png"))
        {
            if (stream == null)
                throw new InvalidOperationException("The embedded Polyfitty Studio splash image is missing.");
            using (Image image = Image.FromStream(stream))
                BackgroundImage = new Bitmap(image);
        }
    }

    private void OnTimerTick(object sender, EventArgs e)
    {
        _phase = (_phase + 13) % 620;
        _progress.Phase = _phase;
        _progress.Invalidate();

        if (!_payloadStarted)
        {
            _payloadStarted = true;
            BeginPayloadExtraction();
            return;
        }

        bool payloadFinished;
        string payloadRoot;
        string payloadError;
        string payloadStatus;
        lock (_payloadSync)
        {
            payloadFinished = _payloadFinished;
            payloadRoot = _payloadRoot;
            payloadError = _payloadError;
            payloadStatus = _payloadStatus;
        }

        if (!String.IsNullOrEmpty(payloadStatus) && !String.Equals(_status.Text, payloadStatus, StringComparison.Ordinal))
            _status.Text = payloadStatus;

        if (payloadFinished && !_runtimeStarted)
        {
            if (!String.IsNullOrEmpty(payloadError))
            {
                ShowFailure("Could not prepare the embedded Polyfitty Studio application: " + payloadError);
                return;
            }
            _root = payloadRoot;
            _runtimePath = Path.Combine(_root, ".runtime", "python312", "python.exe");
            _startupScript = Path.Combine(_root, "installer", "start_app.ps1");
            _runtimeStarted = true;
            StartRuntime();
            return;
        }

        if (_readyEvent.WaitOne(0))
            ReadStatus();
        ReadStatus();
        if (_process != null && SafeHasExited(_process))
        {
            int code = SafeExitCode(_process);
            if (code == 0)
            {
                ExitCode = 0;
                _allowClose = true;
                Close();
            }
            else
            {
                ShowFailure("Studio startup failed (exit code " + code.ToString() + "). Check the startup logs under LocalAppData\\PolyfittyStudio\\Portable.");
            }
        }
    }

    private void BeginPayloadExtraction()
    {
        ThreadPool.QueueUserWorkItem(delegate(object ignored)
        {
            try
            {
                string root = EmbeddedPortablePayload.Extract(delegate(string message)
                {
                    lock (_payloadSync) _payloadStatus = message;
                });
                lock (_payloadSync)
                {
                    _payloadRoot = root;
                    _payloadFinished = true;
                    _payloadStatus = "Embedded application ready. Preparing the GPU check...";
                }
            }
            catch (Exception ex)
            {
                lock (_payloadSync)
                {
                    _payloadError = ex.Message;
                    _payloadFinished = true;
                }
            }
        });
    }

    private void StartRuntime()
    {
        try
        {
            if (!File.Exists(_runtimePath) || !File.Exists(_startupScript))
                throw new FileNotFoundException("The single-file payload was extracted, but its embedded Python or startup helper is missing.");

            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = _powerShellPath;
            start.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File " +
                QuoteArgument(_startupScript) + " -RootPath " + QuoteArgument(_root) + " -PortablePackage";
            start.WorkingDirectory = _root;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.WindowStyle = ProcessWindowStyle.Hidden;
            start.EnvironmentVariables["POLYFITTY_STARTUP_READY_EVENT"] = _readyEventName;
            start.EnvironmentVariables["POLYFITTY_STARTUP_STATUS_FILE"] = _statusPath;
            _process = Process.Start(start);
            if (_process == null)
                throw new InvalidOperationException("The embedded portable startup helper could not be created.");
        }
        catch (Exception ex)
        {
            ShowFailure("Unable to start the Polyfitty Studio runtime: " + ex.Message);
        }
    }

    private static string QuoteArgument(string value)
    {
        return "\"" + (value ?? String.Empty).Replace("\"", "\\\"") + "\"";
    }

    private void ReadStatus()
    {
        try
        {
            if (!File.Exists(_statusPath)) return;
            string value;
            using (FileStream stream = new FileStream(_statusPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (StreamReader reader = new StreamReader(stream))
                value = reader.ReadToEnd().Trim();
            if (value.Length == 0) return;
            if (String.Equals(value, "Polyfitty Studio is ready.", StringComparison.Ordinal))
            {
                _status.Text = "Studio window is ready · finalizing startup...";
                return;
            }
            if (value.StartsWith("ERROR:", StringComparison.OrdinalIgnoreCase))
            {
                ShowFailure(value.Substring(6).Trim());
                return;
            }
            if (!_failureShown && !String.Equals(_status.Text, value, StringComparison.Ordinal))
                _status.Text = value;
        }
        catch { }
    }

    private void ShowFailure(string message)
    {
        if (_failureShown) return;
        _failureShown = true;
        _timer.Stop();
        _progress.Visible = false;
        _status.Location = new Point(282, 310);
        _status.Size = new Size(423, 48);
        _status.ForeColor = Color.FromArgb(255, 190, 190);
        _status.Text = message;
        _closeButton.Visible = true;
        _closeButton.BringToFront();
    }

    private static bool SafeHasExited(Process process)
    {
        try { return process.HasExited; } catch { return true; }
    }

    private static int SafeExitCode(Process process)
    {
        try { return process.ExitCode; } catch { return 1; }
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        _timer.Stop();
        _timer.Dispose();
        if (_process != null) _process.Dispose();
        base.OnFormClosed(e);
    }
}

internal sealed class MarqueeBar : Control
{
    public int Phase { get; set; }

    public MarqueeBar()
    {
        SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.SupportsTransparentBackColor, true);
        BackColor = Color.Transparent;
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        Rectangle track = new Rectangle(0, 1, Width - 1, Height - 2);
        using (SolidBrush brush = new SolidBrush(Color.FromArgb(8, 22, 44)))
            e.Graphics.FillRectangle(brush, track);

        int segmentWidth = Math.Max(72, Width / 4);
        int x = (Phase % (Width + segmentWidth)) - segmentWidth;
        Rectangle segment = new Rectangle(x, 1, segmentWidth, Height - 2);
        GraphicsState state = e.Graphics.Save();
        e.Graphics.SetClip(track);
        using (LinearGradientBrush gradient = new LinearGradientBrush(
            segment,
            Color.FromArgb(45, 196, 255),
            Color.FromArgb(166, 111, 255),
            LinearGradientMode.Horizontal))
            e.Graphics.FillRectangle(gradient, segment);
        e.Graphics.Restore(state);
    }
}
