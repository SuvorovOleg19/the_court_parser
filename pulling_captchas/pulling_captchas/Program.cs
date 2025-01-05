using System.Text.RegularExpressions;
using System.Text;

namespace pulling_captchas
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            var httpClient = new HttpClient();

            // Указываем путь к папке для хранения капч
            string captchaFolderPath = "captchas";

            // Проверяем, существует ли папка, если нет, создаём её
            if (!Directory.Exists(captchaFolderPath))
            {
                Directory.CreateDirectory(captchaFolderPath);
            }

            for (var i = 0; i < 500; i++)
            {
                var response = await httpClient.GetAsync("https://kamensky--svd.sudrf.ru/modules.php?name=sud_delo&name_op=sf&delo_id=1540005");
                var responseBody = await response.Content.ReadAsStringAsync();

                Regex regex = new Regex("data: image/png;base64,\\s+(\\S+)\\\"");

                var captchaExist = regex.IsMatch(responseBody);

                if (captchaExist)
                {
                    var match = regex.Match(responseBody);

                    var imageBase64 = match.Groups[1].Value;

                    byte[] bytes = Convert.FromBase64String(imageBase64);

                    // Сохраняем файл в указанную папку
                    string filePath = Path.Combine(captchaFolderPath, $"{i}.png");
                    File.WriteAllBytes(filePath, bytes);
                }

                Thread.Sleep(31000);
            }
        }
    }
}
