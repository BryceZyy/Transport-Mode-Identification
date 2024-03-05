import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class JoinJob {

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: JoinJob <input path> <output path>");
            System.exit(-1);
        }

        // Configuration
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Data Join");
        job.setJarByClass(JoinJob.class);
        
        // Set Mapper and Reducer
        job.setMapperClass(JoinMapper.class);
        job.setReducerClass(JoinReducer.class);

        // Define Mapper output key and value types
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        // Define final output key and value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Set input and output formats
        job.setInputFormatClass(TextInputFormat.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Exit
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
