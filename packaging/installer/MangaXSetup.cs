// MangaX Setup: detects the graphics card, downloads the matching portable
// package from the latest GitHub release, and unpacks it.
//
// Built with the C# 5 compiler that ships with .NET Framework 4.x (csc.exe),
// so the language level is deliberately old: no string interpolation, no "?.".
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Management;
using System.Net;
using System.Reflection;
using System.Security.AccessControl;
using System.Security.Cryptography;
using System.Security.Principal;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

[assembly: AssemblyTitle("MangaX Setup")]
[assembly: AssemblyProduct("MangaX")]
[assembly: AssemblyDescription("Downloads and installs the MangaX package that matches this computer.")]
[assembly: AssemblyVersion("1.0.0.0")]

namespace MangaXSetup
{
    internal static class Text2
    {
        public static bool Thai = true;

        public static string T(string thai, string english)
        {
            return Thai ? thai : english;
        }

        public static string Size(long bytes)
        {
            double gigabytes = bytes / 1073741824.0;
            if (gigabytes >= 1.0)
            {
                return gigabytes.ToString("0.00", CultureInfo.InvariantCulture) + " GB";
            }
            return (bytes / 1048576.0).ToString("0", CultureInfo.InvariantCulture) + " MB";
        }
    }

    internal static class Log
    {
        private static readonly object Gate = new object();
        public static string FilePath = Path.Combine(Path.GetTempPath(), "MangaX-Setup.log");

        public static void Write(string message)
        {
            lock (Gate)
            {
                try
                {
                    File.AppendAllText(
                        FilePath,
                        DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture) + " " + message + Environment.NewLine,
                        Encoding.UTF8);
                }
                catch (IOException)
                {
                    // Logging must never break the installation.
                }
                catch (UnauthorizedAccessException)
                {
                }
            }
        }
    }

    internal sealed class Variant
    {
        public readonly string Id;
        public readonly string LabelThai;
        public readonly string LabelEnglish;

        public Variant(string id, string labelThai, string labelEnglish)
        {
            Id = id;
            LabelThai = labelThai;
            LabelEnglish = labelEnglish;
        }

        public string Label
        {
            get { return Text2.T(LabelThai, LabelEnglish); }
        }

        public static readonly Variant[] All = new Variant[]
        {
            new Variant("cuda13.0", "NVIDIA รุ่นใหม่ (CUDA 13.0)", "NVIDIA, recent driver (CUDA 13.0)"),
            new Variant("cuda12.6", "NVIDIA driver เก่า / GTX 10 (CUDA 12.6)", "NVIDIA, older driver / GTX 10 (CUDA 12.6)"),
            new Variant("rocm7.2.1", "AMD Radeon (ROCm 7.2.1)", "AMD Radeon (ROCm 7.2.1)"),
            new Variant("cpu", "ไม่ใช้การ์ดจอ (CPU)", "No graphics card (CPU)"),
        };
    }

    internal sealed class Detection
    {
        public string VariantId = "cpu";
        public string GpuName = "";
        public string Note = "";
    }

    internal static class Hardware
    {
        // AMD cards that the Windows ROCm 7.2.1 PyTorch build supports.
        private static readonly string[] RocmKeywords = new string[]
        {
            "MI300A", "MI300X", "MI350X", "MI355X",
            "RX 7900 XTX", "7900 XTX", "RX 7800 XT", "7800 XT", "RX 7700S", "7700S",
            "FRAMEWORK LAPTOP 16", "STRIX HALO", "RX 9060", "RX 9070",
        };

        public static bool AmdSupportsRocm(string gpuName)
        {
            string upper = (gpuName ?? "").ToUpperInvariant();
            foreach (string keyword in RocmKeywords)
            {
                if (upper.Contains(keyword))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Picks the NVIDIA package from the driver's CUDA ceiling and the card
        /// architecture. Returns null when the driver is too old for any package.
        /// </summary>
        public static string SelectNvidiaVariant(string gpuName, int cudaMajor, Version computeCapability)
        {
            string name = Regex.Replace((gpuName ?? "").ToUpperInvariant(), @"\s+", " ");
            bool series50 = Regex.IsMatch(name, @"\bRTX\s*50\d{2}\b");
            bool series10 = Regex.IsMatch(name, @"\b(?:GTX|GT)\s*10\d{2}\b");
            if (series50)
            {
                // RTX 50 is not supported by the CUDA 12.6 build.
                return cudaMajor >= 13 ? "cuda13.0" : null;
            }
            if (cudaMajor < 12)
            {
                return null;
            }
            if (cudaMajor == 12 || series10)
            {
                return "cuda12.6";
            }
            // CUDA 13 dropped everything older than Turing (compute capability 7.5).
            if (computeCapability != null && computeCapability >= new Version(7, 5))
            {
                return "cuda13.0";
            }
            return "cuda12.6";
        }

        public static List<string> VideoControllerNames()
        {
            List<string> names = new List<string>();
            try
            {
                using (ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT Name FROM Win32_VideoController"))
                {
                    foreach (ManagementBaseObject item in searcher.Get())
                    {
                        string name = Convert.ToString(item["Name"], CultureInfo.InvariantCulture);
                        if (!string.IsNullOrWhiteSpace(name))
                        {
                            names.Add(name.Trim());
                        }
                    }
                }
            }
            catch (Exception error)
            {
                Log.Write("WMI video controller query failed: " + error.Message);
            }
            return names;
        }

        private static string RunTool(string fileName, string arguments)
        {
            try
            {
                ProcessStartInfo info = new ProcessStartInfo(fileName, arguments);
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                info.RedirectStandardOutput = true;
                using (Process process = Process.Start(info))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    if (!process.WaitForExit(15000))
                    {
                        process.Kill();
                        return null;
                    }
                    return process.ExitCode == 0 ? output : null;
                }
            }
            catch (Exception error)
            {
                Log.Write(fileName + " unavailable: " + error.Message);
                return null;
            }
        }

        public static Detection Detect()
        {
            Detection result = new Detection();
            List<string> names = VideoControllerNames();
            Log.Write("Video controllers: " + string.Join(" | ", names.ToArray()));

            string nvidia = null;
            string amd = null;
            foreach (string name in names)
            {
                string upper = name.ToUpperInvariant();
                if (nvidia == null && (upper.Contains("NVIDIA") || upper.Contains("GEFORCE") || upper.Contains("QUADRO")))
                {
                    nvidia = name;
                }
                else if (upper.Contains("AMD") || upper.Contains("RADEON"))
                {
                    // Prefer a card the ROCm build supports over an integrated one.
                    if (amd == null || AmdSupportsRocm(name))
                    {
                        amd = name;
                    }
                }
            }

            if (nvidia != null)
            {
                result.GpuName = nvidia;
                DetectNvidia(result);
                return result;
            }
            if (amd != null)
            {
                result.GpuName = amd;
                if (AmdSupportsRocm(amd))
                {
                    result.VariantId = "rocm7.2.1";
                }
                else
                {
                    result.Note = Text2.T(
                        "การ์ด AMD รุ่นนี้ยังไม่รองรับการเร่งด้วย GPU จึงใช้ชุด CPU",
                        "This AMD card has no GPU acceleration support, so the CPU package is used.");
                }
                return result;
            }
            result.GpuName = names.Count > 0 ? names[0] : Text2.T("ไม่พบการ์ดจอ", "No graphics card found");
            return result;
        }

        private static void DetectNvidia(Detection result)
        {
            string summary = RunTool("nvidia-smi", "");
            string query = RunTool("nvidia-smi", "--query-gpu=name,compute_cap --format=csv,noheader,nounits");
            if (summary == null)
            {
                result.Note = Text2.T(
                    "ไม่พบ driver NVIDIA จึงใช้ชุด CPU ติดตั้ง driver แล้วเปิดตัวติดตั้งนี้อีกครั้งเพื่อใช้การ์ดจอ",
                    "No NVIDIA driver was found, so the CPU package is used. Install the driver and run this setup again to use the card.");
                return;
            }

            // Old and new nvidia-smi headers: "CUDA Version: 12.8" / "CUDA UMD Version: 13.4".
            Match cuda = Regex.Match(summary, @"CUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)");
            int cudaMajor = cuda.Success ? int.Parse(cuda.Groups[1].Value, CultureInfo.InvariantCulture) : 0;

            Version best = null;
            string bestName = null;
            if (query != null)
            {
                foreach (string line in query.Split(new char[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                {
                    int comma = line.LastIndexOf(',');
                    Version capability;
                    if (comma > 0 && Version.TryParse(line.Substring(comma + 1).Trim(), out capability))
                    {
                        if (best == null || capability > best)
                        {
                            best = capability;
                            bestName = line.Substring(0, comma).Trim();
                        }
                    }
                }
            }
            if (bestName != null)
            {
                result.GpuName = bestName;
            }
            Log.Write("NVIDIA: " + result.GpuName + ", CUDA major " + cudaMajor + ", compute capability " + best);

            string variant = SelectNvidiaVariant(result.GpuName, cudaMajor, best);
            if (variant == null)
            {
                result.Note = Text2.T(
                    "driver NVIDIA เก่าเกินไป จึงใช้ชุด CPU อัปเดต driver แล้วเปิดตัวติดตั้งนี้อีกครั้งเพื่อใช้การ์ดจอ",
                    "The NVIDIA driver is too old, so the CPU package is used. Update the driver and run this setup again to use the card.");
                return;
            }
            result.VariantId = variant;
        }
    }

    internal sealed class Asset
    {
        public string Name;
        public string Url;
        public long Size;
        public string Sha256;
    }

    internal sealed class Release
    {
        public string Tag;
        public List<Asset> Assets = new List<Asset>();

        /// <summary>Returns the ordered archive volumes of one package, or an empty list.</summary>
        public List<Asset> PartsFor(string variantId)
        {
            Regex pattern = new Regex("^MangaX-" + Regex.Escape(variantId) + @"-v.+\.7z\.(\d{3})$");
            SortedDictionary<int, Asset> ordered = new SortedDictionary<int, Asset>();
            foreach (Asset asset in Assets)
            {
                Match match = pattern.Match(asset.Name);
                if (match.Success)
                {
                    ordered[int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture)] = asset;
                }
            }
            List<Asset> parts = new List<Asset>();
            int expected = 1;
            foreach (KeyValuePair<int, Asset> entry in ordered)
            {
                if (entry.Key != expected)
                {
                    // A missing volume makes the whole package unusable.
                    return new List<Asset>();
                }
                parts.Add(entry.Value);
                expected++;
            }
            return parts;
        }

        public long TotalSize(string variantId)
        {
            long total = 0;
            foreach (Asset part in PartsFor(variantId))
            {
                total += part.Size;
            }
            return total;
        }
    }

    internal static class GitHub
    {
        public const string Repository = "nousx/MangaX";
        private const string UserAgent = "MangaX-Setup";

        public static HttpWebRequest Request(string url)
        {
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
            request.UserAgent = UserAgent;
            request.Timeout = 30000;
            request.ReadWriteTimeout = 60000;
            request.AllowAutoRedirect = true;
            return request;
        }

        public static Release Parse(string json)
        {
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            serializer.MaxJsonLength = int.MaxValue;
            IDictionary<string, object> root = serializer.DeserializeObject(json) as IDictionary<string, object>;
            if (root == null || !root.ContainsKey("tag_name"))
            {
                throw new InvalidDataException("Unexpected release response.");
            }
            Release release = new Release();
            release.Tag = Convert.ToString(root["tag_name"], CultureInfo.InvariantCulture);
            IEnumerable assets = root.ContainsKey("assets") ? root["assets"] as IEnumerable : null;
            if (assets != null)
            {
                foreach (object item in assets)
                {
                    IDictionary<string, object> entry = item as IDictionary<string, object>;
                    if (entry == null)
                    {
                        continue;
                    }
                    Asset asset = new Asset();
                    asset.Name = Convert.ToString(entry["name"], CultureInfo.InvariantCulture);
                    asset.Url = Convert.ToString(entry["browser_download_url"], CultureInfo.InvariantCulture);
                    asset.Size = Convert.ToInt64(entry["size"], CultureInfo.InvariantCulture);
                    object digest;
                    if (entry.TryGetValue("digest", out digest) && digest != null)
                    {
                        string value = Convert.ToString(digest, CultureInfo.InvariantCulture);
                        if (value.StartsWith("sha256:", StringComparison.OrdinalIgnoreCase))
                        {
                            asset.Sha256 = value.Substring(7).ToLowerInvariant();
                        }
                    }
                    release.Assets.Add(asset);
                }
            }
            return release;
        }

        public static Release Latest()
        {
            HttpWebRequest request = Request("https://api.github.com/repos/" + Repository + "/releases/latest");
            request.Accept = "application/vnd.github+json";
            using (WebResponse response = request.GetResponse())
            using (StreamReader reader = new StreamReader(response.GetResponseStream(), Encoding.UTF8))
            {
                return Parse(reader.ReadToEnd());
            }
        }
    }

    internal sealed class InstallOptions
    {
        public string VariantId;
        public string Directory;
        public bool DesktopShortcut = true;
    }

    internal sealed class Installer
    {
        private const int MaxAttempts = 6;
        private readonly Release release;
        private readonly InstallOptions options;
        private readonly CancellationToken cancel;

        // (status text, percent 0-100 or -1 for "busy")
        public Action<string, int> Progress = delegate { };

        public Installer(Release release, InstallOptions options, CancellationToken cancel)
        {
            this.release = release;
            this.options = options;
            this.cancel = cancel;
        }

        public void Run()
        {
            List<Asset> parts = release.PartsFor(options.VariantId);
            if (parts.Count == 0)
            {
                throw new InvalidOperationException(Text2.T(
                    "ไม่พบไฟล์ของชุดนี้ในเวอร์ชันล่าสุด",
                    "The latest release has no files for this package."));
            }
            foreach (Asset part in parts)
            {
                // Never unpack a file that cannot be checked against the release.
                if (string.IsNullOrEmpty(part.Sha256))
                {
                    throw new InvalidDataException(Text2.T(
                        "เวอร์ชันนี้ไม่มีค่า SHA-256 สำหรับตรวจสอบไฟล์ จึงไม่ติดตั้ง",
                        "This release publishes no SHA-256 digest to verify the files, so it will not be installed."));
                }
            }
            PrepareDirectory(options.Directory);
            string package = Path.Combine(options.Directory, "MangaX-package.7z.part");
            Log.Write("Installing " + release.Tag + " " + options.VariantId + " into " + options.Directory);

            Download(parts, package);
            Verify(parts, package);
            Extract(package);
            File.Delete(package);
            File.Delete(package + ".info");
            if (options.DesktopShortcut)
            {
                CreateShortcut();
            }
            Log.Write("Installation finished.");
        }

        /// <summary>
        /// Creates the install folder so that only this user, administrators and
        /// the system can write to it. Folders made directly under a drive root
        /// otherwise inherit write access for every local account, which would
        /// let another account replace the program files.
        /// </summary>
        private static void PrepareDirectory(string path)
        {
            SecurityIdentifier user = WindowsIdentity.GetCurrent().User;
            SecurityIdentifier administrators = new SecurityIdentifier(WellKnownSidType.BuiltinAdministratorsSid, null);
            SecurityIdentifier system = new SecurityIdentifier(WellKnownSidType.LocalSystemSid, null);
            if (!Directory.Exists(path))
            {
                DirectorySecurity security = new DirectorySecurity();
                security.SetAccessRuleProtection(true, false);
                foreach (SecurityIdentifier sid in new SecurityIdentifier[] { user, administrators, system })
                {
                    security.AddAccessRule(new FileSystemAccessRule(
                        sid,
                        FileSystemRights.FullControl,
                        InheritanceFlags.ContainerInherit | InheritanceFlags.ObjectInherit,
                        PropagationFlags.None,
                        AccessControlType.Allow));
                }
                Directory.CreateDirectory(path, security);
                return;
            }
            // An existing folder keeps its permissions, but it must not belong
            // to someone else who could have prepared it in advance.
            IdentityReference owner = Directory.GetAccessControl(path, AccessControlSections.Owner).GetOwner(typeof(SecurityIdentifier));
            if (!user.Equals(owner) && !administrators.Equals(owner) && !system.Equals(owner))
            {
                throw new UnauthorizedAccessException(Text2.T(
                    "โฟลเดอร์นี้เป็นของบัญชีผู้ใช้อื่น เลือกโฟลเดอร์อื่นเพื่อความปลอดภัย",
                    "This folder is owned by another account. Choose a different folder to stay safe."));
            }
        }

        private void Download(List<Asset> parts, string package)
        {
            long total = 0;
            foreach (Asset part in parts)
            {
                total += part.Size;
            }

            // The marker ties a partial file to one release, so a resumed
            // download never mixes volumes from two versions.
            string marker = release.Tag + "|" + options.VariantId + "|" + total.ToString(CultureInfo.InvariantCulture);
            string markerFile = package + ".info";
            if (File.Exists(package) && (!File.Exists(markerFile) || File.ReadAllText(markerFile) != marker || new FileInfo(package).Length > total))
            {
                File.Delete(package);
            }
            File.WriteAllText(markerFile, marker);

            long start = 0;
            foreach (Asset part in parts)
            {
                int attempt = 0;
                while (true)
                {
                    cancel.ThrowIfCancellationRequested();
                    long have = File.Exists(package) ? new FileInfo(package).Length : 0;
                    if (have >= start + part.Size)
                    {
                        break;
                    }
                    try
                    {
                        DownloadPart(part, package, start, Math.Max(0, have - start), total);
                        attempt = 0;
                    }
                    catch (OperationCanceledException)
                    {
                        throw;
                    }
                    catch (Exception error)
                    {
                        if (!(error is WebException) && !(error is IOException))
                        {
                            throw;
                        }
                        attempt++;
                        Log.Write("Download attempt " + attempt + " failed for " + part.Name + ": " + error.Message);
                        if (attempt >= MaxAttempts)
                        {
                            throw new IOException(Text2.T(
                                "ดาวน์โหลดไม่สำเร็จ ตรวจอินเทอร์เน็ตแล้วกดติดตั้งอีกครั้ง โปรแกรมจะโหลดต่อจากจุดเดิม",
                                "The download failed. Check the connection and press Install again; it resumes where it stopped.")
                                + "\n\n" + error.Message, error);
                        }
                        Progress(Text2.T("การเชื่อมต่อหลุด กำลังลองใหม่...", "Connection lost, retrying..."), -1);
                        cancel.WaitHandle.WaitOne(3000 * attempt);
                    }
                }
                start += part.Size;
            }
        }

        private void DownloadPart(Asset part, string package, long start, long offset, long total)
        {
            HttpWebRequest request = GitHub.Request(part.Url);
            if (offset > 0)
            {
                request.AddRange(offset);
            }
            using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
            {
                if (offset > 0 && response.StatusCode != HttpStatusCode.PartialContent)
                {
                    // The server ignored the range: restart this volume.
                    offset = 0;
                }
                using (Stream input = response.GetResponseStream())
                using (FileStream output = new FileStream(package, FileMode.OpenOrCreate, FileAccess.Write, FileShare.Read))
                {
                    output.SetLength(start + offset);
                    output.Seek(0, SeekOrigin.End);
                    byte[] buffer = new byte[1 << 17];
                    long done = start + offset;
                    long windowBytes = 0;
                    double speed = 0;
                    Stopwatch window = Stopwatch.StartNew();
                    long limit = start + part.Size;
                    int read;
                    while ((read = input.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        cancel.ThrowIfCancellationRequested();
                        if (done + read > limit)
                        {
                            throw new IOException("Server sent more data than the release lists for " + part.Name);
                        }
                        output.Write(buffer, 0, read);
                        done += read;
                        windowBytes += read;
                        if (window.ElapsedMilliseconds >= 500)
                        {
                            speed = windowBytes / window.Elapsed.TotalSeconds;
                            windowBytes = 0;
                            window.Restart();
                            string status = Text2.T("กำลังดาวน์โหลด ", "Downloading ")
                                + Text2.Size(done) + " / " + Text2.Size(total)
                                + "  (" + (speed / 1048576.0).ToString("0.0", CultureInfo.InvariantCulture) + " MB/s)";
                            Progress(status, (int)(done * 100 / total));
                        }
                    }
                    if (done < limit)
                    {
                        throw new IOException("Connection closed early for " + part.Name);
                    }
                }
            }
        }

        private void Verify(List<Asset> parts, string package)
        {
            long total = 0;
            foreach (Asset part in parts)
            {
                total += part.Size;
            }
            long done = 0;
            using (FileStream input = new FileStream(package, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                byte[] buffer = new byte[1 << 20];
                foreach (Asset part in parts)
                {
                    using (SHA256 sha = SHA256.Create())
                    {
                        long remaining = part.Size;
                        while (remaining > 0)
                        {
                            cancel.ThrowIfCancellationRequested();
                            int read = input.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
                            if (read <= 0)
                            {
                                throw new IOException("Downloaded package is shorter than expected.");
                            }
                            sha.TransformBlock(buffer, 0, read, null, 0);
                            remaining -= read;
                            done += read;
                            Progress(Text2.T("กำลังตรวจสอบไฟล์...", "Verifying files..."), (int)(done * 100 / total));
                        }
                        sha.TransformFinalBlock(buffer, 0, 0);
                        string actual = BitConverter.ToString(sha.Hash).Replace("-", "").ToLowerInvariant();
                        if (actual != part.Sha256)
                        {
                            input.Close();
                            File.Delete(package);
                            Log.Write("Digest mismatch for " + part.Name + ": " + actual + " != " + part.Sha256);
                            throw new InvalidDataException(Text2.T(
                                "ไฟล์ที่ดาวน์โหลดเสียหาย ลบทิ้งแล้ว กดติดตั้งอีกครั้งเพื่อโหลดใหม่",
                                "The downloaded file was corrupt and has been removed. Press Install again to download it."));
                        }
                    }
                }
            }
        }

        private static string UnpackExtractor()
        {
            string folder = Path.Combine(Path.GetTempPath(), "MangaX-Setup-" + Process.GetCurrentProcess().Id);
            Directory.CreateDirectory(folder);
            string target = Path.Combine(folder, "7zr.exe");
            using (Stream resource = Assembly.GetExecutingAssembly().GetManifestResourceStream("7zr.exe"))
            {
                if (resource == null)
                {
                    throw new FileNotFoundException("The bundled 7-Zip extractor is missing from this build.");
                }
                using (FileStream output = new FileStream(target, FileMode.Create, FileAccess.Write))
                {
                    resource.CopyTo(output);
                }
            }
            return target;
        }

        private void Extract(string package)
        {
            Progress(Text2.T("กำลังแตกไฟล์...", "Unpacking..."), 0);
            string extractor = UnpackExtractor();
            try
            {
                string target = options.Directory.TrimEnd('\\', '/');
                ProcessStartInfo info = new ProcessStartInfo(
                    extractor,
                    "x \"" + package + "\" \"-o" + target + "\" -y -aoa -bsp1 -bso0");
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                info.RedirectStandardOutput = true;
                info.RedirectStandardError = true;
                StringBuilder errors = new StringBuilder();
                using (Process process = new Process())
                {
                    process.StartInfo = info;
                    process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
                    {
                        if (e.Data != null)
                        {
                            lock (errors)
                            {
                                errors.AppendLine(e.Data);
                            }
                        }
                    };
                    process.Start();
                    process.BeginErrorReadLine();

                    // 7-Zip redraws its progress line in place, so read characters
                    // rather than lines and pick the percentage out of the tail.
                    StringBuilder tail = new StringBuilder();
                    char[] buffer = new char[256];
                    int read;
                    while ((read = process.StandardOutput.Read(buffer, 0, buffer.Length)) > 0)
                    {
                        if (cancel.IsCancellationRequested)
                        {
                            process.Kill();
                            process.WaitForExit();
                            cancel.ThrowIfCancellationRequested();
                        }
                        tail.Append(buffer, 0, read);
                        if (tail.Length > 512)
                        {
                            tail.Remove(0, tail.Length - 256);
                        }
                        MatchCollection matches = Regex.Matches(tail.ToString(), @"(\d{1,3})%");
                        if (matches.Count > 0)
                        {
                            int percent = int.Parse(matches[matches.Count - 1].Groups[1].Value, CultureInfo.InvariantCulture);
                            Progress(Text2.T("กำลังแตกไฟล์...", "Unpacking..."), Math.Min(100, percent));
                        }
                    }
                    process.WaitForExit();
                    // 7-Zip exit code 1 is "finished with warnings".
                    if (process.ExitCode > 1)
                    {
                        Log.Write("7zr exit code " + process.ExitCode + ": " + errors);
                        throw new IOException(Text2.T(
                            "แตกไฟล์ไม่สำเร็จ ตรวจพื้นที่ว่างในดิสก์และสิทธิ์เขียนโฟลเดอร์",
                            "Unpacking failed. Check free disk space and write access to the folder.")
                            + "\n\n" + errors.ToString().Trim());
                    }
                }
            }
            finally
            {
                try
                {
                    Directory.Delete(Path.GetDirectoryName(extractor), true);
                }
                catch (IOException)
                {
                }
                catch (UnauthorizedAccessException)
                {
                }
            }
            if (!File.Exists(Path.Combine(options.Directory, "Win-Start.bat")))
            {
                throw new InvalidDataException("The package did not contain Win-Start.bat.");
            }
        }

        private void CreateShortcut()
        {
            try
            {
                string link = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "MangaX.lnk");
                Type shellType = Type.GetTypeFromProgID("WScript.Shell");
                object shell = Activator.CreateInstance(shellType);
                object shortcut = shellType.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, shell, new object[] { link });
                Type type = shortcut.GetType();
                type.InvokeMember("TargetPath", BindingFlags.SetProperty, null, shortcut, new object[] { Path.Combine(options.Directory, "Win-Start.bat") });
                type.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, shortcut, new object[] { options.Directory });
                string icon = Path.Combine(options.Directory, @"desktop_qt_ui\ui\icons\icon.ico");
                if (File.Exists(icon))
                {
                    type.InvokeMember("IconLocation", BindingFlags.SetProperty, null, shortcut, new object[] { icon });
                }
                type.InvokeMember("Save", BindingFlags.InvokeMethod, null, shortcut, null);
            }
            catch (Exception error)
            {
                // A missing shortcut is not worth failing a finished install.
                Log.Write("Could not create the desktop shortcut: " + error.Message);
            }
        }
    }

    internal sealed class SetupForm : Form
    {
        private readonly Label gpuLabel = new Label();
        private readonly Label noteLabel = new Label();
        private readonly ComboBox variantBox = new ComboBox();
        private readonly TextBox folderBox = new TextBox();
        private readonly Button browseButton = new Button();
        private readonly CheckBox shortcutBox = new CheckBox();
        private readonly ProgressBar progressBar = new ProgressBar();
        private readonly Label statusLabel = new Label();
        private readonly Button installButton = new Button();
        private readonly Button closeButton = new Button();

        private Release release;
        private CancellationTokenSource cancelSource;
        private bool installed;
        public string ScreenshotPath;
        public string PresetVariant;

        public SetupForm(string folder)
        {
            Text = "MangaX Setup";
            Font = new Font("Segoe UI", 9.5f);
            AutoScaleMode = AutoScaleMode.Dpi;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(560, 400);
            Padding = new Padding(18);
            try
            {
                Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
            }
            catch (Exception)
            {
            }

            TableLayoutPanel layout = new TableLayoutPanel();
            layout.Dock = DockStyle.Fill;
            layout.ColumnCount = 2;
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            Controls.Add(layout);

            Label title = new Label();
            title.Text = Text2.T("ติดตั้ง MangaX", "Install MangaX");
            title.Font = new Font("Segoe UI", 16f, FontStyle.Bold);
            title.AutoSize = true;
            title.Margin = new Padding(0, 0, 0, 2);
            AddRow(layout, title);

            Label subtitle = new Label();
            subtitle.Text = Text2.T(
                "ตัวติดตั้งจะเลือกชุดที่ตรงกับการ์ดจอของเครื่องนี้ แล้วดาวน์โหลดให้เอง",
                "Setup picks the package that matches this computer's graphics card and downloads it.");
            subtitle.AutoSize = true;
            subtitle.MaximumSize = new Size(520, 0);
            subtitle.ForeColor = SystemColors.GrayText;
            subtitle.Margin = new Padding(0, 0, 0, 14);
            AddRow(layout, subtitle);

            gpuLabel.Text = Text2.T("กำลังตรวจการ์ดจอ...", "Checking the graphics card...");
            gpuLabel.AutoSize = true;
            gpuLabel.MaximumSize = new Size(520, 0);
            gpuLabel.Margin = new Padding(0, 0, 0, 8);
            AddRow(layout, gpuLabel);

            noteLabel.AutoSize = true;
            noteLabel.MaximumSize = new Size(520, 0);
            noteLabel.ForeColor = Color.FromArgb(176, 96, 0);
            noteLabel.Margin = new Padding(0, 0, 0, 8);
            noteLabel.Visible = false;
            AddRow(layout, noteLabel);

            AddRow(layout, Caption(Text2.T("ชุดที่จะติดตั้ง", "Package")));
            variantBox.DropDownStyle = ComboBoxStyle.DropDownList;
            variantBox.Dock = DockStyle.Fill;
            variantBox.Margin = new Padding(0, 0, 0, 10);
            variantBox.Enabled = false;
            AddRow(layout, variantBox);

            AddRow(layout, Caption(Text2.T("โฟลเดอร์ติดตั้ง", "Install folder")));
            folderBox.Text = folder;
            folderBox.Dock = DockStyle.Fill;
            folderBox.Margin = new Padding(0, 0, 6, 10);
            browseButton.Text = Text2.T("เลือก...", "Browse...");
            browseButton.AutoSize = true;
            browseButton.Margin = new Padding(0, 0, 0, 10);
            browseButton.Click += OnBrowse;
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.Controls.Add(folderBox, 0, layout.RowCount);
            layout.Controls.Add(browseButton, 1, layout.RowCount);
            layout.RowCount++;

            shortcutBox.Text = Text2.T("สร้างทางลัดบนเดสก์ท็อป", "Create a desktop shortcut");
            shortcutBox.Checked = true;
            shortcutBox.AutoSize = true;
            shortcutBox.Margin = new Padding(0, 0, 0, 12);
            AddRow(layout, shortcutBox);

            progressBar.Dock = DockStyle.Fill;
            progressBar.Height = 18;
            progressBar.Margin = new Padding(0, 0, 0, 4);
            AddRow(layout, progressBar);

            statusLabel.AutoSize = true;
            statusLabel.MaximumSize = new Size(520, 0);
            statusLabel.Margin = new Padding(0, 0, 0, 10);
            statusLabel.Text = Text2.T("กำลังตรวจเวอร์ชันล่าสุด...", "Looking up the latest version...");
            AddRow(layout, statusLabel);

            FlowLayoutPanel buttons = new FlowLayoutPanel();
            buttons.FlowDirection = FlowDirection.RightToLeft;
            buttons.Dock = DockStyle.Fill;
            buttons.AutoSize = true;
            closeButton.Text = Text2.T("ปิด", "Close");
            closeButton.AutoSize = true;
            closeButton.MinimumSize = new Size(96, 32);
            closeButton.Click += delegate { Close(); };
            installButton.Text = Text2.T("ติดตั้ง", "Install");
            installButton.AutoSize = true;
            installButton.MinimumSize = new Size(120, 32);
            installButton.Enabled = false;
            installButton.Click += OnInstall;
            buttons.Controls.Add(closeButton);
            buttons.Controls.Add(installButton);
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.Controls.Add(buttons, 0, layout.RowCount);
            layout.SetColumnSpan(buttons, 2);
            layout.RowCount++;

            AcceptButton = installButton;
            Shown += OnShown;
            FormClosing += OnClosing;
        }

        private static Label Caption(string text)
        {
            Label label = new Label();
            label.Text = text;
            label.AutoSize = true;
            label.Margin = new Padding(0, 0, 0, 3);
            return label;
        }

        private static void AddRow(TableLayoutPanel layout, Control control)
        {
            layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            layout.Controls.Add(control, 0, layout.RowCount);
            layout.SetColumnSpan(control, 2);
            layout.RowCount++;
        }

        private void OnShown(object sender, EventArgs e)
        {
            Thread worker = new Thread(delegate()
            {
                Detection detection = Hardware.Detect();
                Release found = null;
                string failure = null;
                try
                {
                    found = GitHub.Latest();
                }
                catch (Exception error)
                {
                    Log.Write("Release lookup failed: " + error);
                    failure = error.Message;
                }
                BeginInvoke(new MethodInvoker(delegate { OnReady(detection, found, failure); }));
            });
            worker.IsBackground = true;
            worker.Start();
        }

        private void OnReady(Detection detection, Release found, string failure)
        {
            gpuLabel.Text = Text2.T("การ์ดจอ: ", "Graphics card: ") + detection.GpuName;
            if (detection.Note.Length > 0)
            {
                noteLabel.Text = detection.Note;
                noteLabel.Visible = true;
            }
            if (found == null)
            {
                statusLabel.Text = Text2.T(
                    "ตรวจเวอร์ชันล่าสุดไม่ได้ ตรวจอินเทอร์เน็ตแล้วเปิดใหม่อีกครั้ง\n",
                    "Could not look up the latest version. Check the connection and start again.\n") + failure;
                return;
            }
            release = found;
            string wanted = PresetVariant ?? detection.VariantId;
            foreach (Variant variant in Variant.All)
            {
                long size = found.TotalSize(variant.Id);
                if (size == 0)
                {
                    continue;
                }
                string label = variant.Label + "  ·  " + Text2.Size(size);
                if (variant.Id == detection.VariantId)
                {
                    label += Text2.T("  (แนะนำ)", "  (recommended)");
                }
                variantBox.Items.Add(new VariantItem(variant.Id, label));
                if (variant.Id == wanted)
                {
                    variantBox.SelectedIndex = variantBox.Items.Count - 1;
                }
            }
            if (variantBox.Items.Count == 0)
            {
                statusLabel.Text = Text2.T(
                    "เวอร์ชันล่าสุดยังไม่มีไฟล์ติดตั้ง",
                    "The latest release has no installable packages yet.");
                return;
            }
            if (variantBox.SelectedIndex < 0)
            {
                variantBox.SelectedIndex = 0;
            }
            variantBox.Enabled = true;
            installButton.Enabled = true;
            statusLabel.Text = Text2.T("พร้อมติดตั้ง MangaX ", "Ready to install MangaX ") + found.Tag;

            if (ScreenshotPath != null)
            {
                using (Bitmap bitmap = new Bitmap(Width, Height))
                {
                    DrawToBitmap(bitmap, new Rectangle(0, 0, Width, Height));
                    bitmap.Save(ScreenshotPath);
                }
                Close();
            }
        }

        private void OnBrowse(object sender, EventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                dialog.Description = Text2.T("เลือกโฟลเดอร์สำหรับติดตั้ง MangaX", "Choose where to install MangaX");
                if (dialog.ShowDialog(this) == DialogResult.OK)
                {
                    // Never unpack straight into a folder the user already uses.
                    folderBox.Text = Path.Combine(dialog.SelectedPath, "MangaX");
                }
            }
        }

        private bool Confirm(string message)
        {
            return MessageBox.Show(this, message, Text, MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes;
        }

        private void OnInstall(object sender, EventArgs e)
        {
            if (installed)
            {
                Process.Start(new ProcessStartInfo(Path.Combine(folderBox.Text, "Win-Start.bat")) { WorkingDirectory = folderBox.Text });
                Close();
                return;
            }

            string folder;
            try
            {
                folder = Path.GetFullPath(folderBox.Text.Trim());
            }
            catch (Exception)
            {
                MessageBox.Show(this, Text2.T("โฟลเดอร์ติดตั้งไม่ถูกต้อง", "The install folder is not valid."), Text);
                return;
            }
            VariantItem item = (VariantItem)variantBox.SelectedItem;
            long size = release.TotalSize(item.Id);

            if (Regex.IsMatch(folder, @"[^\x00-\x7F]") && !Confirm(Text2.T(
                "โฟลเดอร์มีตัวอักษรที่ไม่ใช่ภาษาอังกฤษ บางส่วนของโปรแกรมอาจทำงานผิดพลาด\nแนะนำให้ใช้โฟลเดอร์ชื่อภาษาอังกฤษ เช่น C:\\MangaX\n\nติดตั้งต่อหรือไม่",
                "The folder contains non-English characters, which can break parts of the app.\nA plain folder such as C:\\MangaX is recommended.\n\nInstall anyway?")))
            {
                return;
            }
            bool existing = File.Exists(Path.Combine(folder, "Win-Start.bat"));
            if (existing && !Confirm(Text2.T(
                "โฟลเดอร์นี้มี MangaX อยู่แล้ว การติดตั้งจะเขียนทับไฟล์โปรแกรมและการตั้งค่าเริ่มต้น\n\nติดตั้งทับหรือไม่",
                "MangaX is already in this folder. Installing overwrites the program files and default settings.\n\nInstall over it?")))
            {
                return;
            }
            if (!existing && Directory.Exists(folder) && Directory.GetFileSystemEntries(folder).Length > 0
                && !File.Exists(Path.Combine(folder, "MangaX-package.7z.part.info"))
                && !Confirm(Text2.T(
                    "โฟลเดอร์นี้มีไฟล์อื่นอยู่แล้ว ไฟล์ของ MangaX จะถูกวางปนกัน\n\nติดตั้งต่อหรือไม่",
                    "This folder already contains other files; MangaX would be unpacked among them.\n\nInstall anyway?")))
            {
                return;
            }
            try
            {
                // Archive plus unpacked files; PyTorch packs to roughly a third.
                long needed = size * 4;
                long free = new DriveInfo(Path.GetPathRoot(folder)).AvailableFreeSpace;
                if (free < needed && !Confirm(Text2.T(
                    "พื้นที่ว่างอาจไม่พอ ต้องใช้ประมาณ " + Text2.Size(needed) + " เหลือ " + Text2.Size(free) + "\n\nติดตั้งต่อหรือไม่",
                    "There may not be enough free space: about " + Text2.Size(needed) + " needed, " + Text2.Size(free) + " free.\n\nInstall anyway?")))
                {
                    return;
                }
            }
            catch (Exception error)
            {
                Log.Write("Free space check skipped: " + error.Message);
            }

            InstallOptions options = new InstallOptions();
            options.VariantId = item.Id;
            options.Directory = folder;
            options.DesktopShortcut = shortcutBox.Checked;
            folderBox.Text = folder;
            SetBusy(true);

            cancelSource = new CancellationTokenSource();
            Installer installer = new Installer(release, options, cancelSource.Token);
            installer.Progress = delegate(string status, int percent)
            {
                BeginInvoke(new MethodInvoker(delegate { ShowProgress(status, percent); }));
            };
            Thread worker = new Thread(delegate()
            {
                Exception failure = null;
                try
                {
                    installer.Run();
                }
                catch (Exception error)
                {
                    failure = error;
                }
                BeginInvoke(new MethodInvoker(delegate { OnFinished(failure); }));
            });
            worker.IsBackground = true;
            worker.Start();
        }

        private void SetBusy(bool busy)
        {
            variantBox.Enabled = !busy;
            folderBox.Enabled = !busy;
            browseButton.Enabled = !busy;
            shortcutBox.Enabled = !busy;
            installButton.Enabled = !busy;
            closeButton.Text = busy ? Text2.T("ยกเลิก", "Cancel") : Text2.T("ปิด", "Close");
        }

        private void ShowProgress(string status, int percent)
        {
            if (cancelSource == null)
            {
                return;
            }
            statusLabel.Text = status;
            if (percent < 0)
            {
                progressBar.Style = ProgressBarStyle.Marquee;
            }
            else
            {
                progressBar.Style = ProgressBarStyle.Continuous;
                progressBar.Value = Math.Max(0, Math.Min(100, percent));
            }
        }

        private void OnFinished(Exception failure)
        {
            cancelSource = null;
            progressBar.Style = ProgressBarStyle.Continuous;
            SetBusy(false);
            if (failure == null)
            {
                installed = true;
                progressBar.Value = 100;
                statusLabel.Text = Text2.T("ติดตั้งเสร็จแล้ว", "Installation finished.");
                variantBox.Enabled = false;
                folderBox.Enabled = false;
                browseButton.Enabled = false;
                shortcutBox.Enabled = false;
                installButton.Text = Text2.T("เปิด MangaX", "Start MangaX");
                return;
            }
            progressBar.Value = 0;
            if (failure is OperationCanceledException)
            {
                statusLabel.Text = Text2.T(
                    "ยกเลิกแล้ว กดติดตั้งอีกครั้งเพื่อโหลดต่อจากจุดเดิม",
                    "Cancelled. Press Install again to resume the download.");
                return;
            }
            Log.Write("Installation failed: " + failure);
            statusLabel.Text = Text2.T("ติดตั้งไม่สำเร็จ", "Installation failed.");
            MessageBox.Show(
                this,
                failure.Message + "\n\n" + Text2.T("บันทึกการทำงาน: ", "Log file: ") + Log.FilePath,
                Text,
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
        }

        private void OnClosing(object sender, FormClosingEventArgs e)
        {
            if (cancelSource != null)
            {
                // First click cancels; the window stays until the worker stops.
                cancelSource.Cancel();
                statusLabel.Text = Text2.T("กำลังยกเลิก...", "Cancelling...");
                e.Cancel = true;
            }
        }

        private sealed class VariantItem
        {
            public readonly string Id;
            private readonly string label;

            public VariantItem(string id, string label)
            {
                Id = id;
                this.label = label;
            }

            public override string ToString()
            {
                return label;
            }
        }
    }

    internal static class Program
    {
        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        private static string Value(string[] args, string name)
        {
            int index = Array.IndexOf(args, name);
            return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
        }

        private static bool PrefersThai()
        {
            if (CultureInfo.CurrentUICulture.Name.StartsWith("th", StringComparison.OrdinalIgnoreCase)
                || CultureInfo.CurrentCulture.Name.StartsWith("th", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
            // Many Thai users run an English Windows with a Thai keyboard layout.
            foreach (InputLanguage language in InputLanguage.InstalledInputLanguages)
            {
                if (language.Culture.Name.StartsWith("th", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        public static string DefaultFolder()
        {
            string drive = Path.GetPathRoot(Environment.GetFolderPath(Environment.SpecialFolder.System)) ?? @"C:\";
            return Path.Combine(drive, "MangaX");
        }

        [STAThread]
        private static int Main(string[] args)
        {
            // GitHub requires TLS 1.2, which .NET Framework 4.5 does not enable by default.
            ServicePointManager.SecurityProtocol |= (SecurityProtocolType)3072;
            Log.FilePath = Value(args, "--log") ?? Log.FilePath;
            string language = Value(args, "--lang");
            Text2.Thai = language == null ? PrefersThai() : language == "th";
            Log.Write("MangaX Setup started: " + string.Join(" ", args));

            if (Array.IndexOf(args, "--self-test") >= 0)
            {
                return SelfTest.Run();
            }
            if (Array.IndexOf(args, "--detect") >= 0)
            {
                Detection detection = Hardware.Detect();
                Log.Write("Detected: " + detection.GpuName + " -> " + detection.VariantId + " " + detection.Note);
                return 0;
            }
            if (Array.IndexOf(args, "--auto") >= 0)
            {
                return RunUnattended(args);
            }

            SetProcessDPIAware();
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            SetupForm form = new SetupForm(Value(args, "--dir") ?? DefaultFolder());
            form.ScreenshotPath = Value(args, "--screenshot");
            form.PresetVariant = Value(args, "--variant");
            Application.Run(form);
            return 0;
        }

        /// <summary>Installs without a window; progress goes to the log file.</summary>
        private static int RunUnattended(string[] args)
        {
            try
            {
                InstallOptions options = new InstallOptions();
                options.VariantId = Value(args, "--variant") ?? Hardware.Detect().VariantId;
                options.Directory = Path.GetFullPath(Value(args, "--dir") ?? DefaultFolder());
                options.DesktopShortcut = Array.IndexOf(args, "--no-shortcut") < 0;
                Installer installer = new Installer(GitHub.Latest(), options, CancellationToken.None);
                int last = -2;
                installer.Progress = delegate(string status, int percent)
                {
                    if (percent != last)
                    {
                        last = percent;
                        Log.Write(percent + "% " + status);
                    }
                };
                installer.Run();
                return 0;
            }
            catch (Exception error)
            {
                Log.Write("Unattended installation failed: " + error);
                return 1;
            }
        }
    }

    internal static class SelfTest
    {
        private static int failures;

        private static void Expect(string name, object expected, object actual)
        {
            if (!object.Equals(expected, actual))
            {
                failures++;
                Log.Write("SELF-TEST FAILED " + name + ": expected " + (expected ?? "null") + ", got " + (actual ?? "null"));
            }
        }

        public static int Run()
        {
            Version ampere = new Version(8, 6);
            Version pascal = new Version(6, 1);
            Expect("RTX 3070, new driver", "cuda13.0", Hardware.SelectNvidiaVariant("NVIDIA GeForce RTX 3070", 13, ampere));
            Expect("RTX 3080, CUDA 12 driver", "cuda12.6", Hardware.SelectNvidiaVariant("NVIDIA GeForce RTX 3080", 12, ampere));
            Expect("RTX 3080, CUDA 11 driver", null, Hardware.SelectNvidiaVariant("NVIDIA GeForce RTX 3080", 11, ampere));
            Expect("GTX 1080 Ti", "cuda12.6", Hardware.SelectNvidiaVariant("NVIDIA GeForce GTX 1080 Ti", 13, pascal));
            Expect("RTX 5090", "cuda13.0", Hardware.SelectNvidiaVariant("NVIDIA GeForce RTX 5090", 13, new Version(12, 0)));
            Expect("RTX 5090, CUDA 12 driver", null, Hardware.SelectNvidiaVariant("NVIDIA GeForce RTX 5090", 12, new Version(12, 0)));
            Expect("unknown capability", "cuda12.6", Hardware.SelectNvidiaVariant("NVIDIA Quadro P2000", 13, null));
            Expect("RTX 2060 (Turing)", "cuda13.0", Hardware.SelectNvidiaVariant("NVIDIA GeForce RTX 2060", 13, new Version(7, 5)));
            Expect("RX 9070 XT", true, Hardware.AmdSupportsRocm("AMD Radeon RX 9070 XT"));
            Expect("RX 6600", false, Hardware.AmdSupportsRocm("AMD Radeon RX 6600"));

            Release release = GitHub.Parse(
                "{\"tag_name\":\"v9.9.9\",\"assets\":["
                + "{\"name\":\"MangaX-cuda13.0-v9.9.9.7z.002\",\"browser_download_url\":\"u2\",\"size\":5,\"digest\":\"sha256:AB\"},"
                + "{\"name\":\"MangaX-cuda13.0-v9.9.9.7z.001\",\"browser_download_url\":\"u1\",\"size\":7,\"digest\":null},"
                + "{\"name\":\"MangaX-cuda12.6-v9.9.9.7z.002\",\"browser_download_url\":\"u3\",\"size\":3},"
                + "{\"name\":\"MangaX-cpu-v9.9.9.7z.001\",\"browser_download_url\":\"u4\",\"size\":2},"
                + "{\"name\":\"MangaX-Setup.exe\",\"browser_download_url\":\"u5\",\"size\":1}]}");
            List<Asset> parts = release.PartsFor("cuda13.0");
            Expect("part count", 2, parts.Count);
            Expect("part order", "u1", parts.Count > 0 ? parts[0].Url : null);
            Expect("digest lower-cased", "ab", parts.Count > 1 ? parts[1].Sha256 : null);
            Expect("total size", 12L, release.TotalSize("cuda13.0"));
            Expect("missing first volume", 0, release.PartsFor("cuda12.6").Count);
            Expect("cpu is not matched by cuda", 1, release.PartsFor("cpu").Count);
            Expect("absent package", 0, release.PartsFor("rocm7.2.1").Count);

            Log.Write(failures == 0 ? "SELF-TEST OK" : "SELF-TEST: " + failures + " failure(s)");
            return failures == 0 ? 0 : 1;
        }
    }
}
